"""Chargement du modele de production depuis le registre MLflow.

Aucun HTTP ici : toute la logique delicate (resolution de l'alias, priorite du
seuil, refus propre) vit dans ce module, ou elle est testable sans serveur.
La couche FastAPI viendra se poser par-dessus.

La chaine de resolution :

    models:/fraud-detector@production
            |  get_model_version_by_alias
            v
        ModelVersion  (version, run_id)
            |  get_run
            v
        Run  metrics[frozen_threshold], tags[git_commit, dataset_sha256]
            |
            v
        ModelBundle   <- le modele et son seuil arrivent ENSEMBLE

Mesures sur le modele reel, contre ~8 ms pour une prediction :

    ~13,8 s   tout premier chargement, artefacts pas encore en cache
    ~5,1 s    processus neuf, artefacts deja en cache local
    ~80 ms    rechargement dans le MEME processus

Un conteneur qui demarre paie le cas HAUT : en Kubernetes chaque pod part avec
un cache d'artefacts vide. Charger par requete ne serait donc pas "lent", ce
serait inutilisable -- de 600 a 1700 fois le cout d'une prediction.

D'ou un chargement unique au demarrage, et un service qui demarre degrade
plutot que de mourir quand le registre n'est pas joignable : une sonde de
liveness testant le modele tuerait le conteneur avant la fin du chargement,
indefiniment.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
    PRODUCTION_MODEL_URI,
    REGISTERED_MODEL,
)

# Levier metier : permet d'ajuster le point de fonctionnement sans reentrainer.
THRESHOLD_ENV_VAR = "FRAUD_THRESHOLD"

# Metrique posee par src/training/tracking.py sur le run qui a produit la version.
THRESHOLD_METRIC = "frozen_threshold"


class ModelLoadError(RuntimeError):
    """Le modele de production n'a pas pu etre charge.

    Le step 2.B attrapera cette erreur pour demarrer degrade : le service vit,
    /ready repond 503 avec la raison, et Kubernetes se contente de ne pas lui
    router de trafic.
    """


@dataclass(frozen=True)
class ModelBundle:
    """Le modele charge et TOUT ce qui doit voyager avec lui.

    Un seul objet plutot que des variables separees : chaque reponse de
    /predict devra porter la version et le seuil utilises. S'ils vivaient
    separement, rien ne garantirait qu'ils correspondent au modele reellement
    en memoire.
    """

    model: Any = field(repr=False)
    model_name: str
    version: str
    run_id: str
    threshold: float
    threshold_source: str  # "env" ou "mlflow_run"
    feature_columns: tuple[str, ...]
    model_uri: str
    loaded_at: str
    model_kind: str
    git_commit: str
    git_dirty: str
    dataset_sha256: str
    test_pr_auc: float | None

    def describe(self) -> dict[str, Any]:
        """Vue serialisable, destinee a /ready et aux logs de demarrage.

        threshold_source y figure explicitement : une variable d'environnement
        oubliee d'un deploiement precedent doit rester VISIBLE, jamais
        silencieuse.
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "run_id": self.run_id,
            "model_kind": self.model_kind,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "n_features": len(self.feature_columns),
            "model_uri": self.model_uri,
            "loaded_at": self.loaded_at,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "dataset_sha256": self.dataset_sha256,
            "test_pr_auc": self.test_pr_auc,
        }


def build_client(tracking_uri: str | None = None) -> MlflowClient:
    """Client MLflow pointant sur le registre configure.

    MLFLOW_TRACKING_URI est lu par src/config.py depuis l'environnement : en
    Phase 5, le pod Kubernetes visera un serveur distant sans changement de code.
    """
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def resolve_production_version(
    client: MlflowClient,
    *,
    name: str = REGISTERED_MODEL,
    alias: str = PRODUCTION_ALIAS,
) -> Any:
    """Resout l'alias en une version precise du registre."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        raise ModelLoadError(
            f"Aucun modele sous l'alias @{alias} pour {name!r} dans le registre.\n"
            "  Entraine puis promeus un modele :\n"
            "      python -m src.training.train --promote\n"
            f"  Registre interroge : {mlflow.get_tracking_uri()}\n"
            f"  Detail MLflow : {exc}"
        ) from exc


def resolve_threshold(run: Any, *, env: Mapping[str, str] | None = None) -> tuple[float, str]:
    """Determine le seuil de decision et d'ou il vient.

    Ordre : variable d'environnement, puis metrique du run, puis REFUS.

    L'environnement passe en premier parce qu'un operateur qui pose
    FRAUD_THRESHOLD le fait deliberement : c'est le levier metier qui justifie
    que le seuil vive cote service et non dans le modele.

    Le refus final est volontaire. Un scoreur de fraude avec un seuil arbitraire
    est PIRE qu'un scoreur absent : il repond normalement, avec une precision
    qui n'est pas celle annoncee, et n'alerte personne.
    """
    environment = os.environ if env is None else env
    raw = environment.get(THRESHOLD_ENV_VAR)

    if raw is not None and raw.strip():
        try:
            value = float(raw)
        except ValueError as exc:
            raise ModelLoadError(
                f"{THRESHOLD_ENV_VAR}={raw!r} n'est pas un nombre."
            ) from exc
        if not 0.0 <= value <= 1.0:
            raise ModelLoadError(
                f"{THRESHOLD_ENV_VAR}={value} est hors de [0, 1]. Un seuil se "
                "compare a une probabilite : 80 n'est pas 0.80."
            )
        return value, "env"

    metric = run.data.metrics.get(THRESHOLD_METRIC)
    if metric is not None:
        return float(metric), "mlflow_run"

    raise ModelLoadError(
        f"Le run {run.info.run_id} ne porte pas la metrique {THRESHOLD_METRIC!r}, "
        f"et {THRESHOLD_ENV_VAR} n'est pas definie.\n"
        "  Refus de servir : un seuil arbitraire donnerait une fausse assurance.\n"
        f"  Pose {THRESHOLD_ENV_VAR}, ou promeus une version dont le run porte "
        "son seuil (une cible de precision inatteignable n'en produit pas)."
    )


def extract_feature_columns(model: Any) -> tuple[str, ...]:
    """Lit les colonnes attendues dans la SIGNATURE du modele charge.

    Deliberement PAS depuis src/features/build.py. FEATURE_COLUMNS decrit ce que
    le code ACTUEL produit ; la signature decrit ce sur quoi le modele DEPLOYE a
    ete entraine. Ajouter une variable au pipeline sans reentrainer ferait
    diverger les deux, et le service construirait des DataFrames que le modele
    n'attend pas.

    Le service s'adapte au modele qu'on lui donne, il ne suppose pas.
    """
    metadata = getattr(model, "metadata", None)
    schema = metadata.get_input_schema() if metadata is not None else None
    if schema is None:
        raise ModelLoadError(
            "Le modele charge n'a pas de signature d'entree : impossible de "
            "savoir quelles colonnes il attend.\n"
            "  Reentraine via src.training.tracking, qui enregistre la signature."
        )

    names = tuple(schema.input_names())
    if not names:
        raise ModelLoadError("La signature d'entree du modele ne declare aucune colonne.")
    return names


def load_production_model(
    *,
    client: MlflowClient | None = None,
    model_uri: str = PRODUCTION_MODEL_URI,
    name: str = REGISTERED_MODEL,
    alias: str = PRODUCTION_ALIAS,
    env: Mapping[str, str] | None = None,
    loader: Callable[[str], Any] | None = None,
) -> ModelBundle:
    """Charge le modele de production et tout son contexte.

    ``loader`` est injectable pour les tests : charger le vrai modele coute
    ~13,8 s, ce qui rendrait la suite inutilisable et exigerait une base MLflow
    que la CI n'aura pas.
    """
    active_client = client if client is not None else build_client()

    version = resolve_production_version(active_client, name=name, alias=alias)
    run = active_client.get_run(version.run_id)

    # Le seuil est resolu AVANT le chargement : c'est un controle a quelques
    # microsecondes, alors que le chargement coute ~13,8 s. Echouer vite sur une
    # configuration incomplete plutot qu'apres quatorze secondes de travail
    # destine a etre jete.
    threshold, threshold_source = resolve_threshold(run, env=env)

    load = loader if loader is not None else mlflow.pyfunc.load_model
    try:
        model = load(model_uri)
    except Exception as exc:
        raise ModelLoadError(
            f"Chargement de {model_uri} impossible : {type(exc).__name__}: {exc}"
        ) from exc

    tags = run.data.tags
    return ModelBundle(
        model=model,
        model_name=name,
        version=str(version.version),
        run_id=str(version.run_id),
        threshold=threshold,
        threshold_source=threshold_source,
        feature_columns=extract_feature_columns(model),
        model_uri=model_uri,
        loaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        model_kind=str(run.data.params.get("model_kind", "unknown")),
        git_commit=str(tags.get("git_commit", "unknown")),
        git_dirty=str(tags.get("git_dirty", "unknown")),
        dataset_sha256=str(tags.get("dataset_sha256", "unknown")),
        test_pr_auc=(
            float(run.data.metrics["test_pr_auc"])
            if "test_pr_auc" in run.data.metrics
            else None
        ),
    )
