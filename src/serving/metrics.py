"""Metriques Prometheus du service de scoring.

Prometheus fonctionne en **pull** : toutes les 15 a 30 s il fait un GET sur
/metrics et lit un texte brut. Le service n'envoie donc rien ; il expose un etat
courant, et c'est Prometheus qui vient le chercher.

Consequence directe : **/metrics ne doit jamais echouer**, meme en mode degrade.
Sinon Prometheus perd le service de vue exactement au moment ou l'on a le plus
besoin de le voir. Ce module ne depend donc d'aucun modele charge.

Histogram et non Summary, partout ou l'on veut des percentiles. Les deux savent
les calculer, mais un Summary les calcule DANS le processus : avec 3 replicas on
obtiendrait 3 p99, et la moyenne de trois p99 n'est pas le p99 global -- c'est un
nombre qui ne veut rien dire. Un Histogram expose des TRANCHES, que Prometheus
additionne entre pods avant de calculer le percentile. C'est le seul des deux qui
donne un vrai p99 global.

Cardinalite : jamais d'etiquette a valeurs illimitees (transaction_id,
user_id...). Chaque combinaison cree une serie temporelle, et Prometheus
s'effondre. Ici : decision (2 valeurs) x endpoint (2 valeurs) = 4 series au plus
par metrique.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    Info,
    generate_latest,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.serving.model import ModelBundle

# Les tranches PAR DEFAUT (0.005, 0.01, ... 10.0) visent des latences HTTP en
# secondes. Notre prediction mesure ~8 ms : presque tout tomberait dans une
# seule tranche et les percentiles seraient inexploitables. On resserre donc
# autour de la milliseconde, en gardant des tranches hautes pour voir une
# degradation.
LATENCY_BUCKETS = (
    0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020,
    0.050, 0.100, 0.250, 0.500, 1.000, float("inf"),
)

# Le score vit dans [0, 1] : les tranches par defaut, qui montent a 10 secondes,
# n'auraient aucun sens. On met de la resolution AUX DEUX EXTREMITES -- la masse
# des transactions normales est ecrasee pres de 0, et c'est pres de 1 que se
# trouve l'information. 0.8 encadre la region du seuil (0.80801547).
#
# Les tranches sont figees a la declaration, donc AVANT que le modele et son
# seuil soient charges : impossible d'en derive une du seuil courant.
SCORE_BUCKETS = (
    0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5,
    0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
)

# Cles d'etiquettes de fraud_model_info : elles doivent rester identiques d'un
# appel a l'autre, d'ou un jeu de valeurs de repli en mode degrade.
UNKNOWN_MODEL_INFO = {
    "version": "none",
    "model_kind": "none",
    "run_id": "none",
    "threshold": "none",
    "threshold_source": "none",
    "git_commit": "none",
    "dataset_sha256": "none",
}

# prometheus_client ajoute lui-meme le suffixe _total au compteur : la metrique
# exposee s'appelle fraud_predictions_total.
PREDICTIONS = Counter(
    "fraud_predictions",
    "Nombre de transactions scorees, par decision et par endpoint.",
    ["decision", "endpoint"],
)

LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Duree du scoring par le modele, hors surcout HTTP et validation.",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)

SCORES = Histogram(
    "fraud_score",
    (
        "Distribution des scores de fraude. Signal de derive : le score est une "
        "projection en une dimension des 30 variables d'entree, donc un "
        "deplacement de cette distribution signale un changement du trafic."
    ),
    buckets=SCORE_BUCKETS,
)

MODEL_INFO = Info(
    "fraud_model",
    "Version du modele reellement servie, et provenance de son seuil.",
)
MODEL_INFO.info(UNKNOWN_MODEL_INFO)


def observe_prediction(*, endpoint: str, score: float, is_fraud: bool) -> None:
    """Enregistre une transaction scoree.

    Le compteur s'incremente PAR TRANSACTION, y compris dans un lot : c'est ce
    qui donne un vrai debit en transactions/seconde. Le nombre d'appels reste
    lisible separement via fraud_prediction_latency_seconds_count.
    """
    PREDICTIONS.labels(
        decision="fraud" if is_fraud else "legitimate",
        endpoint=endpoint,
    ).inc()
    SCORES.observe(score)


@contextmanager
def measure_latency(endpoint: str) -> Iterator[None]:
    """Chronometre le scoring lui-meme.

    Volontairement autour du seul appel au modele : la metrique s'appelle
    "prediction latency", elle doit donc mesurer la prediction. Le surcout HTTP
    et la validation Pydantic sont mesurables separement, par un middleware, si
    le besoin s'en fait sentir.

    ``finally`` : une prediction qui echoue doit quand meme etre chronometree,
    sinon les incidents disparaissent des percentiles -- precisement les cas
    qu'on veut voir.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)


def set_model_info(bundle: ModelBundle | None) -> None:
    """Publie la version servie, ou l'etat degrade.

    Metrique "info" : une jauge toujours a 1 dont l'interet est dans ses
    ETIQUETTES. Elle permet de correler, sur un tableau de bord, un changement
    de latence ou de distribution avec un changement de version.

    En mode degrade on publie version="none" plutot que rien : une metrique
    absente ne se distingue pas d'un service injoignable, alors qu'une valeur
    explicite est alertable.
    """
    if bundle is None:
        MODEL_INFO.info(UNKNOWN_MODEL_INFO)
        return

    MODEL_INFO.info(
        {
            "version": str(bundle.version),
            "model_kind": str(bundle.model_kind),
            "run_id": str(bundle.run_id),
            "threshold": f"{bundle.threshold:.8f}",
            "threshold_source": str(bundle.threshold_source),
            "git_commit": str(bundle.git_commit),
            "dataset_sha256": str(bundle.dataset_sha256),
        }
    )


def render_exposition() -> tuple[bytes, str]:
    """Rend l'exposition texte et son Content-Type.

    Le registre par defaut apporte gratuitement python_gc_* et python_info.

    Note pour la Phase 5 : prometheus_client tient ses compteurs en memoire, PAR
    PROCESSUS. Avec `uvicorn --workers 4`, un scrape tomberait sur un worker au
    hasard et ne verrait qu'un quart du trafic. En Kubernetes ce n'est pas un
    probleme : on met a l'echelle par replicas, et Prometheus interroge chaque
    pod separement. Donc un seul worker par conteneur.
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
