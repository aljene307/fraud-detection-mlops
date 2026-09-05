"""Tests de l'API de scoring (step 2.B).

Aucun test ne charge le vrai modele. Deux mecanismes le permettent :

* ``TestClient(app)`` SANS gestionnaire de contexte n'execute PAS le lifespan.
  L'application demarre donc sans modele -- ce qui est exactement l'etat degrade
  qu'on veut eprouver.
* ``with TestClient(app)`` execute le lifespan ; on remplace alors
  ``load_production_model`` par une doublure pour tester le lifespan lui-meme.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving import app as app_module
from src.serving.app import app, check_schema_matches_model
from src.serving.model import ModelBundle, ModelLoadError
from src.serving.schemas import (
    EXAMPLE_TRANSACTION,
    MAX_BATCH_SIZE,
    TRANSACTION_FIELDS,
)

THRESHOLD = 0.8080154657363892


class FakeModel:
    """Renvoie un score fixe, et retient les DataFrames recus."""

    def __init__(self, score: float = 0.95) -> None:
        self.score = score
        self.seen: list[Any] = []

    def predict(self, frame: Any) -> np.ndarray:
        self.seen.append(frame)
        n = len(frame)
        # Deux colonnes, comme pyfunc_predict_fn="predict_proba".
        return np.column_stack([np.full(n, 1.0 - self.score), np.full(n, self.score)])


def make_bundle(score: float = 0.95, threshold: float = THRESHOLD) -> ModelBundle:
    return ModelBundle(
        model=FakeModel(score),
        model_name="fraud-detector",
        version="2",
        run_id="9add346914be",
        threshold=threshold,
        threshold_source="mlflow_run",
        feature_columns=TRANSACTION_FIELDS,
        model_uri="models:/fraud-detector@production",
        loaded_at="2026-09-05T10:00:00+00:00",
        model_kind="xgb",
        git_commit="c3b4ef6",
        git_dirty="false",
        dataset_sha256="76274b69",
        test_pr_auc=0.8727,
    )


@pytest.fixture
def loaded_client() -> TestClient:
    """Application avec un modele en place, sans passer par le lifespan."""
    client = TestClient(app)
    app.state.bundle = make_bundle()
    app.state.load_error = None
    yield client
    app.state.bundle = None
    app.state.load_error = None


@pytest.fixture
def degraded_client() -> TestClient:
    """Application demarree en mode degrade : aucun modele."""
    client = TestClient(app)
    app.state.bundle = None
    app.state.load_error = "Aucun modele sous l'alias @production."
    yield client
    app.state.load_error = None


# ===========================================================================
# /health : le test le plus important du module
# ===========================================================================


def test_health_is_ok_when_the_model_is_loaded(loaded_client: TestClient) -> None:
    response = loaded_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_stays_ok_when_the_model_failed_to_load(
    degraded_client: TestClient,
) -> None:
    """LE test qui protege la decision "demarrage degrade".

    Un conteneur dont le chargement a echoue est parfaitement VIVANT : il attend
    que MLflow revienne. Si /health repondait autre chose que 200, la sonde de
    liveness le tuerait, il redemarrerait, echouerait a nouveau -- le
    CrashLoopBackOff que toute cette conception vise a eviter.
    """
    response = degraded_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_never_touches_the_bundle(degraded_client: TestClient) -> None:
    """Meme sans aucun etat sur l'application, /health doit repondre."""
    delattr(app.state, "bundle")
    try:
        assert degraded_client.get("/health").status_code == 200
    finally:
        app.state.bundle = None


# ===========================================================================
# /ready : la sonde de readiness
# ===========================================================================


def test_ready_returns_the_model_description(loaded_client: TestClient) -> None:
    response = loaded_client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model"]["version"] == "2"
    assert body["model"]["threshold"] == pytest.approx(THRESHOLD)
    assert body["model"]["threshold_source"] == "mlflow_run"


def test_ready_is_503_with_the_reason_when_degraded(
    degraded_client: TestClient,
) -> None:
    """503 et non 500 : le service n'est pas casse, il n'est pas encore pret.
    Et la raison doit y figurer, sinon on debug a l'aveugle."""
    response = degraded_client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "@production" in body["reason"]
    assert body["model"] is None


# ===========================================================================
# /predict : le cas nominal
# ===========================================================================


def test_predict_returns_score_decision_and_provenance(
    loaded_client: TestClient,
) -> None:
    response = loaded_client.post("/predict", json=EXAMPLE_TRANSACTION)

    assert response.status_code == 200
    body = response.json()
    assert body["fraud_probability"] == pytest.approx(0.95)
    assert body["is_fraud"] is True
    assert body["threshold"] == pytest.approx(THRESHOLD)
    assert body["threshold_source"] == "mlflow_run"
    assert body["model_name"] == "fraud-detector"
    assert body["model_version"] == "2"


def test_predict_below_threshold_is_not_fraud() -> None:
    client = TestClient(app)
    app.state.bundle = make_bundle(score=0.10)
    try:
        body = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
        assert body["fraud_probability"] == pytest.approx(0.10)
        assert body["is_fraud"] is False
    finally:
        app.state.bundle = None


def test_threshold_comparison_is_inclusive() -> None:
    """Convention >=, la meme que confusion_at_threshold dans metrics.py, avec
    laquelle le seuil a ete choisi. Un > ferait diverger le service de la
    precision annoncee."""
    client = TestClient(app)
    app.state.bundle = make_bundle(score=THRESHOLD)
    try:
        assert client.post("/predict", json=EXAMPLE_TRANSACTION).json()["is_fraud"] is True
    finally:
        app.state.bundle = None


def test_predict_builds_the_dataframe_from_the_model_signature(
    loaded_client: TestClient,
) -> None:
    """Les colonnes viennent de bundle.feature_columns, dans cet ordre."""
    loaded_client.post("/predict", json=EXAMPLE_TRANSACTION)

    frame = app.state.bundle.model.seen[0]
    assert list(frame.columns) == list(TRANSACTION_FIELDS)
    assert len(frame) == 1


# ===========================================================================
# /predict : validation Pydantic (422, faute du client)
# ===========================================================================


def test_missing_field_is_422_and_names_it(loaded_client: TestClient) -> None:
    """422 et non 500 : la faute est cote client, il merite de savoir laquelle."""
    payload = dict(EXAMPLE_TRANSACTION)
    del payload["V14"]

    response = loaded_client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "V14" in response.text


def _post_raw(client: TestClient, payload: dict) -> Any:
    """Envoie le corps JSON tel quel, sans passer par le serialiseur du client.

    NaN et Infinity n'existent PAS dans le JSON standard, et httpx refuse de les
    serialiser. Un client correct ne peut donc pas en envoyer -- mais un client
    artisanal, si : json.dumps de Python emet le litteral non standard `NaN`, et
    json.loads le relit sans broncher. C'est ce chemin-la que allow_inf_nan=False
    garde, et c'est donc celui qu'il faut eprouver.
    """
    return client.post(
        "/predict",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def test_nan_is_rejected(loaded_client: TestClient) -> None:
    """allow_inf_nan=False. Un NaN traverserait sinon jusqu'au modele et
    produirait un score sans valeur, sans lever la moindre erreur."""
    payload = dict(EXAMPLE_TRANSACTION)
    payload["V3"] = float("nan")

    assert _post_raw(loaded_client, payload).status_code == 422


def test_infinity_is_rejected(loaded_client: TestClient) -> None:
    payload = dict(EXAMPLE_TRANSACTION)
    payload["V3"] = float("inf")

    assert _post_raw(loaded_client, payload).status_code == 422


@pytest.mark.parametrize("hour", [-1.0, 24.0, 47.0])
def test_hour_of_day_outside_range_is_rejected(
    loaded_client: TestClient, hour: float
) -> None:
    """Attrape un modulo oublie cote client."""
    payload = dict(EXAMPLE_TRANSACTION)
    payload["hour_of_day"] = hour

    assert loaded_client.post("/predict", json=payload).status_code == 422


def test_negative_log_amount_is_rejected(loaded_client: TestClient) -> None:
    """log1p d'un montant positif est toujours >= 0."""
    payload = dict(EXAMPLE_TRANSACTION)
    payload["log_amount"] = -1.0

    assert loaded_client.post("/predict", json=payload).status_code == 422


def test_raw_amount_instead_of_log_amount_is_caught(
    loaded_client: TestClient,
) -> None:
    """L'erreur client la plus probable. extra="forbid" la rend explicite :
    le message signale a la fois le champ manquant et le champ en trop."""
    payload = dict(EXAMPLE_TRANSACTION)
    del payload["log_amount"]
    payload["Amount"] = 149.62

    response = loaded_client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "log_amount" in response.text
    assert "Amount" in response.text


def test_extreme_pca_values_are_accepted(loaded_client: TestClient) -> None:
    """Aucune borne sur V1-V28 : une borne arbitraire rejetterait les
    transactions extremes, or c'est precisement la que vit la fraude."""
    payload = dict(EXAMPLE_TRANSACTION)
    payload["V14"] = -120.0

    assert loaded_client.post("/predict", json=payload).status_code == 200


# ===========================================================================
# /predict/batch
# ===========================================================================


def test_batch_scores_every_row(loaded_client: TestClient) -> None:
    payload = {"transactions": [EXAMPLE_TRANSACTION] * 5}

    response = loaded_client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    assert len(body["predictions"]) == 5
    assert [item["index"] for item in body["predictions"]] == [0, 1, 2, 3, 4]
    assert body["model_version"] == "2"


def test_batch_calls_the_model_once_for_the_whole_lot(
    loaded_client: TestClient,
) -> None:
    """La raison d'etre du lot : le cout est FIXE par appel (~8 ms pour 1 ligne
    comme pour 200). Un appel par ligne annulerait tout le benefice."""
    loaded_client.post("/predict/batch", json={"transactions": [EXAMPLE_TRANSACTION] * 50})

    model = app.state.bundle.model
    assert len(model.seen) == 1
    assert len(model.seen[0]) == 50


def test_batch_rejects_an_empty_list(loaded_client: TestClient) -> None:
    assert loaded_client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_batch_rejects_more_than_the_cap(loaded_client: TestClient) -> None:
    """Le plafond borne la taille du corps de requete et le travail par appel."""
    payload = {"transactions": [EXAMPLE_TRANSACTION] * (MAX_BATCH_SIZE + 1)}

    assert loaded_client.post("/predict/batch", json=payload).status_code == 422


# ===========================================================================
# Etat degrade : 503 sur le scoring, jamais de plantage
# ===========================================================================


def test_predict_is_503_when_the_model_is_absent(degraded_client: TestClient) -> None:
    response = degraded_client.post("/predict", json=EXAMPLE_TRANSACTION)

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "model_not_loaded"


def test_batch_is_503_when_the_model_is_absent(degraded_client: TestClient) -> None:
    response = degraded_client.post(
        "/predict/batch", json={"transactions": [EXAMPLE_TRANSACTION]}
    )

    assert response.status_code == 503


def test_degraded_503_carries_the_reason(degraded_client: TestClient) -> None:
    body = degraded_client.post("/predict", json=EXAMPLE_TRANSACTION).json()

    assert "@production" in body["detail"]["reason"]


# ===========================================================================
# Le lifespan lui-meme
# ===========================================================================


def test_lifespan_loads_the_model_and_exposes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = make_bundle()
    monkeypatch.setattr(app_module, "load_production_model", lambda: bundle)

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert app.state.bundle is bundle
        assert app.state.load_error is None


def test_lifespan_catches_model_load_error_and_starts_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le mecanisme du demarrage degrade.

    uvicorn fait sys.exit(STARTUP_FAILURE) si le lifespan laisse echapper une
    exception. Attraper ModelLoadError A L'INTERIEUR est donc precisement ce qui
    fait vivre le service au lieu de le tuer.
    """

    def failing_loader() -> ModelBundle:
        raise ModelLoadError("Aucun modele sous l'alias @production.")

    monkeypatch.setattr(app_module, "load_production_model", failing_loader)

    with TestClient(app) as client:
        assert app.state.bundle is None
        assert "@production" in app.state.load_error
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=EXAMPLE_TRANSACTION).status_code == 503


def test_lifespan_lets_unexpected_errors_kill_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seule ModelLoadError est attrapee. Un bug de programmation doit tuer le
    processus, pas se deguiser en indisponibilite temporaire."""

    def buggy_loader() -> ModelBundle:
        raise TypeError("bug de programmation")

    monkeypatch.setattr(app_module, "load_production_model", buggy_loader)

    with pytest.raises(TypeError), TestClient(app):
        pass


# ===========================================================================
# Le controle de coherence contrat <-> signature
# ===========================================================================


def test_schema_check_passes_when_contract_matches_signature() -> None:
    check_schema_matches_model(make_bundle())  # ne doit pas lever


def test_schema_check_reports_columns_on_both_sides() -> None:
    """Le message doit dire ce qui manque ET ce qui est en trop, sinon on ne
    sait pas de quel cote corriger."""
    bundle = make_bundle()
    mismatched = ModelBundle(
        **{
            **{f.name: getattr(bundle, f.name) for f in bundle.__dataclass_fields__.values()},
            "feature_columns": ("V1", "V2", "nouvelle_variable"),
        }
    )

    with pytest.raises(ModelLoadError) as err:
        check_schema_matches_model(mismatched)

    message = str(err.value)
    assert "nouvelle_variable" in message  # attendue par le modele
    assert "V28" in message  # declaree par le contrat, inconnue du modele


def test_schema_mismatch_starts_degraded_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SchemaMismatchError herite de ModelLoadError, donc le meme except
    l'attrape : une incoherence de schema est un modele inutilisable."""
    bundle = make_bundle()
    broken = ModelBundle(
        **{
            **{f.name: getattr(bundle, f.name) for f in bundle.__dataclass_fields__.values()},
            "feature_columns": ("V1",),
        }
    )
    monkeypatch.setattr(app_module, "load_production_model", lambda: broken)

    with TestClient(app) as client:
        assert app.state.bundle is None
        assert "contrat d'API" in app.state.load_error
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


# ===========================================================================
# Le contrat statique face au pipeline actuel
# ===========================================================================


def test_contract_matches_the_current_feature_pipeline() -> None:
    """Le contrat est fige dans le code, mais il doit refleter le pipeline
    d'aujourd'hui. Ce test tombe le jour ou l'un des deux bouge sans l'autre --
    en developpement, avant que le controle au demarrage ne s'en charge."""
    from src.features.build import FEATURE_COLUMNS

    assert set(TRANSACTION_FIELDS) == set(FEATURE_COLUMNS)
    assert len(TRANSACTION_FIELDS) == 30


def test_example_transaction_is_a_valid_payload(loaded_client: TestClient) -> None:
    """L'exemple affiche dans /docs doit reellement fonctionner : un exemple
    faux est pire que pas d'exemple."""
    assert loaded_client.post("/predict", json=EXAMPLE_TRANSACTION).status_code == 200


def test_openapi_documents_every_field() -> None:
    schema = app.openapi()["components"]["schemas"]["TransactionIn"]

    assert set(schema["properties"]) == set(TRANSACTION_FIELDS)
    assert schema.get("additionalProperties") is False  # extra="forbid"
