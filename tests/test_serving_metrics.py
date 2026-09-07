"""Tests de l'exposition Prometheus (step 2.C).

Les metriques prometheus_client sont des singletons de module : les remettre a
zero entre les tests serait fragile et masquerait des interactions. Ces tests
mesurent donc des ECARTS -- on lit la valeur avant, on agit, on relit -- ce qui
les rend independants de l'ordre d'execution.

Ils lisent la sortie texte reelle via le parseur officiel, et non les attributs
internes des objets : c'est exactement ce que Prometheus verra.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from src.serving import metrics as metrics_module
from src.serving.app import app
from src.serving.metrics import (
    LATENCY_BUCKETS,
    SCORE_BUCKETS,
    measure_latency,
    render_exposition,
    set_model_info,
)
from src.serving.schemas import EXAMPLE_TRANSACTION
from tests.test_serving_app import THRESHOLD, make_bundle


def scrape(client: TestClient) -> str:
    """Recupere l'exposition telle que Prometheus la lirait."""
    response = client.get("/metrics")
    assert response.status_code == 200
    return response.text


def sample(text: str, name: str, **labels: str) -> float | None:
    """Extrait la valeur d'un echantillon du texte d'exposition."""
    for family in text_string_to_metric_families(text):
        for item in family.samples:
            if item.name != name:
                continue
            if all(item.labels.get(key) == value for key, value in labels.items()):
                return item.value
    return None


def value(text: str, name: str, **labels: str) -> float:
    """Comme sample(), mais 0.0 quand la serie n'existe pas encore."""
    found = sample(text, name, **labels)
    return 0.0 if found is None else found


@pytest.fixture
def loaded_client() -> TestClient:
    client = TestClient(app)
    app.state.bundle = make_bundle()
    app.state.load_error = None
    yield client
    app.state.bundle = None
    app.state.load_error = None


@pytest.fixture
def degraded_client() -> TestClient:
    client = TestClient(app)
    app.state.bundle = None
    app.state.load_error = "Aucun modele sous l'alias @production."
    yield client
    app.state.load_error = None


# ===========================================================================
# L'endpoint lui-meme
# ===========================================================================


def test_metrics_uses_the_prometheus_content_type(loaded_client: TestClient) -> None:
    """Du texte, pas du JSON. Un mauvais Content-Type et Prometheus ignore la
    reponse."""
    response = loaded_client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=" in response.headers["content-type"]


def test_metrics_answers_even_in_degraded_mode(degraded_client: TestClient) -> None:
    """Comme /health. Prometheus fonctionne en pull : si /metrics echouait sans
    modele, on perdrait le service de vue au pire moment."""
    response = degraded_client.get("/metrics")

    assert response.status_code == 200
    assert "fraud_predictions_total" in response.text


def test_exposition_parses_and_declares_our_metrics(loaded_client: TestClient) -> None:
    names = {family.name for family in text_string_to_metric_families(scrape(loaded_client))}

    assert "fraud_predictions" in names
    assert "fraud_prediction_latency_seconds" in names
    assert "fraud_score" in names
    assert "fraud_model_info" in names


def test_registry_brings_process_metrics_for_free(loaded_client: TestClient) -> None:
    assert "python_info" in scrape(loaded_client)


# ===========================================================================
# Le compteur
# ===========================================================================


def test_prediction_counter_increases_by_one(loaded_client: TestClient) -> None:
    before = value(
        scrape(loaded_client), "fraud_predictions_total",
        decision="fraud", endpoint="single",
    )

    loaded_client.post("/predict", json=EXAMPLE_TRANSACTION)

    after = value(
        scrape(loaded_client), "fraud_predictions_total",
        decision="fraud", endpoint="single",
    )
    assert after - before == pytest.approx(1.0)


def test_counter_separates_fraud_from_legitimate() -> None:
    """L'etiquette decision est ce qui permet de suivre le RATIO d'alertes :
    si 30 % des transactions sont marquees fraude, quelque chose ne va pas."""
    client = TestClient(app)
    app.state.bundle = make_bundle(score=0.01)  # sous le seuil
    try:
        before = value(
            scrape(client), "fraud_predictions_total",
            decision="legitimate", endpoint="single",
        )
        client.post("/predict", json=EXAMPLE_TRANSACTION)
        after = value(
            scrape(client), "fraud_predictions_total",
            decision="legitimate", endpoint="single",
        )
        assert after - before == pytest.approx(1.0)
    finally:
        app.state.bundle = None


def test_batch_counts_every_transaction_not_every_call(
    loaded_client: TestClient,
) -> None:
    """Le compteur s'incremente PAR TRANSACTION : c'est ce qui donne un debit en
    transactions/seconde. Le nombre d'appels reste lisible separement, via
    fraud_prediction_latency_seconds_count."""
    before_rows = value(
        scrape(loaded_client), "fraud_predictions_total",
        decision="fraud", endpoint="batch",
    )
    before_calls = value(
        scrape(loaded_client), "fraud_prediction_latency_seconds_count",
        endpoint="batch",
    )

    loaded_client.post("/predict/batch", json={"transactions": [EXAMPLE_TRANSACTION] * 7})

    text = scrape(loaded_client)
    assert value(text, "fraud_predictions_total", decision="fraud", endpoint="batch") - before_rows == pytest.approx(7.0)
    assert value(text, "fraud_prediction_latency_seconds_count", endpoint="batch") - before_calls == pytest.approx(1.0)


def test_label_cardinality_stays_bounded(loaded_client: TestClient) -> None:
    """Le piege n^1 de Prometheus. decision (2) x endpoint (2) = 4 series au
    plus. Une etiquette a valeurs illimitees ferait exploser le nombre de
    series temporelles."""
    loaded_client.post("/predict", json=EXAMPLE_TRANSACTION)
    loaded_client.post("/predict/batch", json={"transactions": [EXAMPLE_TRANSACTION]})

    series = [
        item
        for family in text_string_to_metric_families(scrape(loaded_client))
        for item in family.samples
        if item.name == "fraud_predictions_total"
    ]
    assert len(series) <= 4
    assert {item.labels["endpoint"] for item in series} <= {"single", "batch"}
    assert {item.labels["decision"] for item in series} <= {"fraud", "legitimate"}


# ===========================================================================
# L'histogramme de latence
# ===========================================================================


def test_latency_is_observed_on_each_call(loaded_client: TestClient) -> None:
    before = value(
        scrape(loaded_client), "fraud_prediction_latency_seconds_count",
        endpoint="single",
    )

    loaded_client.post("/predict", json=EXAMPLE_TRANSACTION)

    after = value(
        scrape(loaded_client), "fraud_prediction_latency_seconds_count",
        endpoint="single",
    )
    assert after - before == pytest.approx(1.0)


def test_latency_buckets_are_tuned_to_milliseconds() -> None:
    """Les tranches par defaut (0.005, 0.01, ... 10.0) visent des latences HTTP
    en secondes. Notre prediction mesure ~8 ms : presque tout tomberait dans une
    seule tranche et les percentiles seraient inexploitables."""
    below_ten_ms = [bound for bound in LATENCY_BUCKETS if bound <= 0.010]

    assert len(below_ten_ms) >= 5
    assert 0.008 in LATENCY_BUCKETS  # la latence mesuree tombe sur une bordure


def test_latency_is_recorded_even_when_scoring_fails() -> None:
    """`finally` dans measure_latency : un incident doit apparaitre dans les
    percentiles, sinon on perd exactement les cas qu'on veut voir."""
    before = value(render_exposition()[0].decode(), "fraud_prediction_latency_seconds_count", endpoint="essai")

    with pytest.raises(RuntimeError), measure_latency("essai"):
        raise RuntimeError("panne du modele")

    after = value(render_exposition()[0].decode(), "fraud_prediction_latency_seconds_count", endpoint="essai")
    assert after - before == pytest.approx(1.0)


# ===========================================================================
# L'histogramme des scores : le signal de derive
# ===========================================================================


def test_score_histogram_records_the_observed_score(
    loaded_client: TestClient,
) -> None:
    before = value(scrape(loaded_client), "fraud_score_sum")

    loaded_client.post("/predict", json=EXAMPLE_TRANSACTION)

    after = value(scrape(loaded_client), "fraud_score_sum")
    assert after - before == pytest.approx(0.95)  # le score de la doublure


def test_score_buckets_are_cumulative(loaded_client: TestClient) -> None:
    """Mecanique de l'histogramme : `le` signifie less-than-or-equal, et chaque
    tranche compte TOUT ce qui est en dessous. C'est ce qui rend l'addition
    entre pods possible -- et donc le vrai p99 global."""
    loaded_client.post("/predict", json=EXAMPLE_TRANSACTION)
    text = scrape(loaded_client)

    counts = [value(text, "fraud_score_bucket", le=str(float(bound))) for bound in SCORE_BUCKETS]
    assert counts == sorted(counts)  # monotone croissant


def test_score_buckets_span_zero_to_one_with_resolution_at_both_ends() -> None:
    """Le score vit dans [0, 1] : des tranches montant a 10 secondes n'auraient
    aucun sens. La masse des transactions normales est ecrasee pres de 0, et
    l'information est pres de 1."""
    assert max(SCORE_BUCKETS) == 1.0
    assert min(SCORE_BUCKETS) <= 0.001
    assert len([b for b in SCORE_BUCKETS if b <= 0.1]) >= 4  # resolution en bas
    assert len([b for b in SCORE_BUCKETS if b >= 0.8]) >= 4  # et en haut
    assert 0.8 in SCORE_BUCKETS  # encadre la region du seuil (0.80801547)


def test_score_histogram_shape_reveals_a_distribution_shift() -> None:
    """Demonstration du signal de derive.

    On soumet d'abord un trafic "normal" (scores ecrases pres de 0), puis un
    trafic decale (scores eleves). La part des observations sous 0.1 s'effondre
    -- sans qu'aucune etiquette de verite ne soit necessaire.

    C'est exactement ce qu'un tableau de bord Grafana montrerait, et ce
    qu'Evidently ira ensuite expliquer variable par variable en Phase 6.
    """
    text_start = render_exposition()[0].decode()
    low_start = value(text_start, "fraud_score_bucket", le="0.1")
    total_start = value(text_start, "fraud_score_count")

    client = TestClient(app)
    app.state.bundle = make_bundle(score=0.001)
    try:
        for _ in range(20):
            client.post("/predict", json=EXAMPLE_TRANSACTION)
    finally:
        app.state.bundle = None

    text_normal = render_exposition()[0].decode()
    low_normal = value(text_normal, "fraud_score_bucket", le="0.1") - low_start
    total_normal = value(text_normal, "fraud_score_count") - total_start
    assert low_normal / total_normal == pytest.approx(1.0)  # 100 % sous 0.1

    app.state.bundle = make_bundle(score=0.97)
    try:
        for _ in range(20):
            client.post("/predict", json=EXAMPLE_TRANSACTION)
    finally:
        app.state.bundle = None

    text_shifted = render_exposition()[0].decode()
    low_shifted = value(text_shifted, "fraud_score_bucket", le="0.1") - low_start
    total_shifted = value(text_shifted, "fraud_score_count") - total_start

    # La proportion sous 0.1 a chute de 100 % a ~50 % : la derive est visible.
    assert low_shifted / total_shifted == pytest.approx(0.5, abs=0.05)


# ===========================================================================
# fraud_model_info
# ===========================================================================


def test_model_info_publishes_the_served_version(loaded_client: TestClient) -> None:
    """Metrique "info" : une jauge a 1 dont l'interet est dans ses etiquettes.
    Elle permet de correler un changement de latence avec un changement de
    version sur un tableau de bord."""
    set_model_info(make_bundle())

    text = scrape(loaded_client)
    assert value(text, "fraud_model_info", version="2", model_kind="xgb") == 1.0
    assert sample(text, "fraud_model_info", threshold_source="mlflow_run") == 1.0


def test_model_info_is_explicit_when_degraded(degraded_client: TestClient) -> None:
    """version="none" plutot que rien : une metrique absente ne se distingue pas
    d'un service injoignable, alors qu'une valeur explicite est alertable."""
    set_model_info(None)

    assert value(scrape(degraded_client), "fraud_model_info", version="none") == 1.0


def test_model_info_label_keys_are_stable() -> None:
    """Les cles doivent etre identiques entre l'etat charge et degrade, sinon
    prometheus_client leve."""
    set_model_info(make_bundle())
    set_model_info(None)
    set_model_info(make_bundle())

    assert value(render_exposition()[0].decode(), "fraud_model_info", version="2") == 1.0


def test_model_info_carries_the_threshold_used(loaded_client: TestClient) -> None:
    set_model_info(make_bundle())

    text = scrape(loaded_client)
    assert sample(text, "fraud_model_info", threshold=f"{THRESHOLD:.8f}") == 1.0


# ===========================================================================
# Le module isole
# ===========================================================================


def test_render_returns_bytes_and_content_type() -> None:
    payload, content_type = render_exposition()

    assert isinstance(payload, bytes)
    assert content_type.startswith("text/plain")


def test_metrics_module_needs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Le module ne doit dependre d'aucun modele charge : c'est ce qui garantit
    que /metrics repond toujours."""
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        metrics_module.SCORES, "observe", lambda v: observed.setdefault("score", v)
    )

    metrics_module.observe_prediction(endpoint="single", score=0.42, is_fraud=False)

    assert observed["score"] == pytest.approx(0.42)
