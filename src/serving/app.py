"""Service de scoring FastAPI.

    uvicorn src.serving.app:app --reload

FastAPI associe des URL a des fonctions Python ; c'est **uvicorn** qui ouvre le
socket et parle HTTP. D'ou la commande ci-dessus : on lance le serveur en lui
indiquant ou trouver l'application.

Deux mecanismes structurent ce fichier.

**Le lifespan** -- un gestionnaire de contexte asynchrone : ce qui precede le
``yield`` s'execute une fois au demarrage, ce qui suit a l'arret. Verifie dans
uvicorn/server.py : ``startup()`` appelle ``lifespan.startup()`` AVANT
``loop.create_server(...)``. Le modele se charge donc pendant que le port est
encore FERME -- une connexion entrante est refusee, pas mise en attente.

Consequence pour la Phase 5 : pendant les 5 a 14 s de chargement, une sonde
httpGet recoit "connection refused". Une sonde de LIVENESS trop impatiente
tuerait le conteneur en plein chargement, indefiniment. La parade Kubernetes est
un startupProbe, qui suspend liveness et readiness tant que l'application n'a
pas demarre.

**Le demarrage degrade** -- uvicorn fait ``sys.exit(STARTUP_FAILURE)`` si le
lifespan laisse echapper une exception. Demarrer degrade consiste donc a
attraper ModelLoadError A L'INTERIEUR du lifespan. Toute AUTRE exception est
laissee passer volontairement : un bug de programmation doit tuer le processus,
pas se deguiser en indisponibilite temporaire.
"""

from __future__ import annotations

import logging
import math
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.serving.metrics import (
    measure_latency,
    observe_prediction,
    render_exposition,
    set_model_info,
)
from src.serving.model import ModelBundle, ModelLoadError, load_production_model
from src.serving.schemas import (
    TRANSACTION_FIELDS,
    BatchIn,
    BatchOut,
    BatchPredictionOut,
    HealthOut,
    PredictionOut,
    TransactionIn,
)

logger = logging.getLogger(__name__)


class SchemaMismatchError(ModelLoadError):
    """Le contrat d'API et la signature du modele ne concordent pas.

    Herite de ModelLoadError pour etre attrapee par le meme ``except`` : une
    incoherence de schema est, du point de vue du service, un modele
    inutilisable. Le service demarre degrade et l'annonce.
    """


def check_schema_matches_model(bundle: ModelBundle) -> None:
    """Relie le contrat statique a la signature du modele deploye.

    C'est le point de jonction des deux regles opposees : on s'adapte a ce
    qu'on consomme, on reste stable pour ce qu'on expose. Si les deux divergent
    -- une variable ajoutee au pipeline sans reentrainement, par exemple -- on
    refuse de servir plutot que de construire des DataFrames que le modele
    n'attend pas.
    """
    declared = set(TRANSACTION_FIELDS)
    expected = set(bundle.feature_columns)
    if declared == expected:
        return

    missing = sorted(expected - declared)
    extra = sorted(declared - expected)
    raise SchemaMismatchError(
        "Le contrat d'API ne correspond pas a la signature du modele "
        f"{bundle.model_name} v{bundle.version}.\n"
        f"  Attendues par le modele, absentes du contrat : {missing or 'aucune'}\n"
        f"  Declarees par le contrat, inconnues du modele : {extra or 'aucune'}\n"
        "  Mets a jour src/serving/schemas.py, ou promeus un modele compatible."
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modele une fois, avant l'ouverture du socket.

    Charger par requete couterait 5 a 14 s a chaque appel, contre ~8 ms pour une
    prediction : ce ne serait pas "lent", ce serait inutilisable.
    """
    app.state.bundle = None
    app.state.load_error = None

    try:
        bundle = load_production_model()
        check_schema_matches_model(bundle)
    except ModelLoadError as exc:
        # DEGRADE : le processus vit, /health repond 200, /ready explique.
        # Kubernetes se contente de ne pas router de trafic vers ce pod.
        app.state.load_error = str(exc)
        logger.error("Demarrage DEGRADE, aucun modele servi :\n%s", exc)
    else:
        app.state.bundle = bundle
        logger.info("Modele charge : %s", bundle.describe())

    # Publiee dans les deux cas : en mode degrade elle vaut version="none",
    # ce qui est alertable -- alors qu'une metrique absente ne se distingue pas
    # d'un service injoignable.
    set_model_info(app.state.bundle)

    yield

    app.state.bundle = None


app = FastAPI(
    title="Fraud Detection Scorer",
    version="0.1.0",
    description=(
        "Score des transactions par carte. Le modele provient uniquement de "
        "l'alias MLflow models:/fraud-detector@production."
    ),
    lifespan=lifespan,
)


def _json_safe(value: Any) -> Any:
    """Remplace les flottants non finis par leur ecriture textuelle."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Repond 422 meme quand la valeur refusee est NaN ou Infinity.

    Sans ceci, le service PLANTE en annoncant le rejet. La chaine est
    contre-intuitive : Pydantic refuse correctement le NaN, puis FastAPI place
    la valeur fautive dans le detail de la reponse 422, et Starlette serialise
    ce detail avec json.dumps(..., allow_nan=False) -- qui leve.

    Le client recevrait alors un 500 ("notre bug") la ou la faute est la sienne
    et merite un 422. On neutralise donc les valeurs non finies AVANT
    serialisation ; elles apparaissent en clair sous forme de texte ("nan"),
    ce qui reste la meilleure information a renvoyer.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": _json_safe(jsonable_encoder(exc.errors()))},
    )


def get_bundle(request: Request) -> ModelBundle:
    """Dependance partagee par les endpoints qui ont besoin du modele.

    FastAPI l'appelle avant l'endpoint et lui passe le resultat. L'interet est
    de centraliser "recupere le bundle, ou renvoie 503" a un seul endroit au
    lieu de le repeter, et de permettre aux tests de la remplacer via
    ``app.dependency_overrides``.
    """
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        reason = getattr(request.app.state, "load_error", None)
        raise _unavailable(reason)
    return bundle


def _unavailable(reason: str | None) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "status": "model_not_loaded",
            "reason": reason or "Le modele de production n'a pas ete charge.",
        },
    )


# Style recommande par FastAPI depuis 0.95 : la dependance vit dans
# l'annotation plutot que dans la valeur par defaut. Plus lisible, reutilisable,
# et sans appel de fonction dans une valeur par defaut.
BundleDep = Annotated[ModelBundle, Depends(get_bundle)]


def _score(bundle: ModelBundle, rows: Sequence[dict[str, float]]) -> np.ndarray:
    """Construit le DataFrame attendu par le modele et renvoie P(fraude).

    Les colonnes viennent de ``bundle.feature_columns``, c'est-a-dire de la
    SIGNATURE du modele deploye -- jamais de src/features/build.py.

    Le modele a ete enregistre avec ``pyfunc_predict_fn="predict_proba"``, donc
    ``predict()`` renvoie deux colonnes : [P(normal), P(fraude)]. C'est la
    seconde qui nous interesse. Sans ce reglage, on recevrait des etiquettes
    0/1 et le seuil deviendrait inapplicable.
    """
    frame = pd.DataFrame(list(rows), columns=list(bundle.feature_columns))
    proba = np.asarray(bundle.model.predict(frame))
    return proba[:, 1] if proba.ndim == 2 else proba.ravel()


@app.get("/health", response_model=HealthOut, tags=["operations"])
def health() -> HealthOut:
    """Sonde de LIVENESS : le processus repond-il ?

    Ne consulte JAMAIS le modele, et c'est tout l'interet. Un conteneur dont le
    chargement a echoue est parfaitement vivant : il attend que MLflow revienne.
    Si cette sonde testait le modele, Kubernetes le tuerait au lieu de
    simplement le retirer du trafic -- exactement le CrashLoopBackOff qu'on veut
    eviter.
    """
    return HealthOut(status="ok")


@app.get("/ready", tags=["operations"])
def ready(request: Request) -> JSONResponse:
    """Sonde de READINESS : peut-on router du trafic vers ce pod ?

    Lit app.state directement, sans passer par get_bundle : cet endpoint doit
    COMPOSER sa reponse 503 en y mettant la raison, pas se contenter d'en lever
    une.
    """
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "reason": getattr(request.app.state, "load_error", None)
                or "Le modele de production n'a pas ete charge.",
                "model": None,
            },
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready", "reason": None, "model": bundle.describe()},
    )


@app.post("/predict", response_model=PredictionOut, tags=["scoring"])
def predict(
    transaction: TransactionIn,
    bundle: BundleDep,
) -> PredictionOut:
    """Score une transaction.

    Volontairement ``def`` et non ``async def`` : la prediction XGBoost est un
    calcul CPU bloquant. En ``async def`` elle bloquerait la boucle
    d'evenements et le debit s'effondrerait ; en ``def``, FastAPI l'execute
    dans un pool de threads, ce qui est correct pour du travail bloquant.
    """
    with measure_latency("single"):
        probability = float(_score(bundle, [transaction.model_dump()])[0])

    # >= et non > : meme convention que confusion_at_threshold dans
    # training/metrics.py, avec laquelle le seuil a ete choisi.
    is_fraud = probability >= bundle.threshold
    observe_prediction(endpoint="single", score=probability, is_fraud=is_fraud)

    return PredictionOut(
        fraud_probability=probability,
        is_fraud=is_fraud,
        threshold=bundle.threshold,
        threshold_source=bundle.threshold_source,
        model_name=bundle.model_name,
        model_version=bundle.version,
    )


@app.post("/predict/batch", response_model=BatchOut, tags=["scoring"])
def predict_batch(
    payload: BatchIn,
    bundle: BundleDep,
) -> BatchOut:
    """Score une liste de transactions en UN SEUL appel au modele.

    Un appel par transaction couterait ~8 ms chacun ; un seul appel sur N lignes
    coute ~8 ms au total. C'est ce qui permettra au simulateur de la Phase 3 de
    mesurer le modele plutot que la latence reseau.
    """
    rows = [transaction.model_dump() for transaction in payload.transactions]

    # Un seul chronometrage pour l'appel entier : c'est bien la duree d'UN
    # scoring. Le compteur, lui, s'incremente par transaction, ce qui donne un
    # debit en transactions/seconde. Les deux lectures restent disponibles :
    # rate(latency_count) = appels/s, rate(predictions_total) = transactions/s.
    with measure_latency("batch"):
        scores = _score(bundle, rows)

    decisions = [float(score) >= bundle.threshold for score in scores]
    for score, is_fraud in zip(scores, decisions, strict=True):
        observe_prediction(endpoint="batch", score=float(score), is_fraud=is_fraud)

    return BatchOut(
        count=len(rows),
        threshold=bundle.threshold,
        threshold_source=bundle.threshold_source,
        model_name=bundle.model_name,
        model_version=bundle.version,
        predictions=[
            BatchPredictionOut(
                index=index,
                fraud_probability=float(score),
                is_fraud=is_fraud,
            )
            for index, (score, is_fraud) in enumerate(
                zip(scores, decisions, strict=True)
            )
        ],
    )


@app.get("/metrics", tags=["operations"])
def metrics() -> Response:
    """Exposition Prometheus.

    Comme /health, cet endpoint ne consulte JAMAIS le modele. Prometheus
    fonctionne en pull : s'il echouait en mode degrade, on perdrait le service
    de vue exactement au moment ou l'on a le plus besoin de le voir.

    On renvoie une Response brute plutot qu'un JSONResponse : le format
    Prometheus est du texte, pas du JSON, et son Content-Type est precis.
    """
    payload, content_type = render_exposition()
    return Response(content=payload, media_type=content_type)
