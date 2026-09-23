"""Tests du simulateur de flux (step 3.A).

Le debit et la latence ne se testent pas de facon deterministe : ils dependent de
la machine. On teste donc ce qui l'est -- l'ordonnancement, le decoupage, les
statistiques -- et surtout la propriete dont la violation serait grave :
``Class`` ne doit JAMAIS partir dans une requete.

Aucun test ne joint un vrai service : une doublure de ``requests.Session``
enregistre ce qui aurait ete envoye, et ``clock``/``sleep`` sont injectes pour
que l'ordonnancement se verifie sans attendre.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import pytest
import requests

from src.config import TARGET
from src.features.build import FEATURE_COLUMNS, SPLIT_FILES
from src.simulator.replay import (
    MIN_SAMPLES_FOR_PERCENTILES,
    OUTSIDE_DOCKER_WARNING,
    LatencySummary,
    Readiness,
    RunConfig,
    ServerSnapshot,
    SimulatorError,
    Stream,
    build_requests,
    check_readiness,
    histogram_quantile,
    is_outside_docker,
    load_stream,
    percentile,
    planned_offsets,
    read_server_snapshot,
    render_report,
    run_campaign,
    total_transactions,
)

METRICS_TEXT = """\
# HELP fraud_prediction_latency_seconds Duree
# TYPE fraud_prediction_latency_seconds histogram
fraud_prediction_latency_seconds_bucket{endpoint="batch",le="0.01"} 2.0
fraud_prediction_latency_seconds_bucket{endpoint="batch",le="0.02"} 8.0
fraud_prediction_latency_seconds_bucket{endpoint="batch",le="+Inf"} 10.0
fraud_prediction_latency_seconds_count{endpoint="batch"} 10.0
fraud_prediction_latency_seconds_sum{endpoint="batch"} 0.15
fraud_prediction_latency_seconds_count{endpoint="single"} 999.0
fraud_prediction_latency_seconds_sum{endpoint="single"} 99.0
# HELP fraud_predictions Nombre
# TYPE fraud_predictions counter
fraud_predictions_total{decision="fraud",endpoint="batch"} 4.0
fraud_predictions_total{decision="legitimate",endpoint="batch"} 96.0
"""

# Deuxieme releve : les compteurs ont avance de 100 appels et 1,5 s.
# La difference donne donc une moyenne de 15 ms et un p50 de 15 ms.
METRICS_AFTER = METRICS_TEXT.replace(
    'le="0.01"} 2.0', 'le="0.01"} 22.0'
).replace(
    'le="0.02"} 8.0', 'le="0.02"} 88.0'
).replace(
    'le="+Inf"} 10.0', 'le="+Inf"} 110.0'
).replace(
    '_count{endpoint="batch"} 10.0', '_count{endpoint="batch"} 110.0'
).replace(
    '_sum{endpoint="batch"} 0.15', '_sum{endpoint="batch"} 1.65'
)


# ---------------------------------------------------------------------------
# Doublures
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status
        self.text = text

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class RecordingSession:
    """Enregistre chaque requete au lieu de l'envoyer.

    Le score renvoye est deterministe : les indices pairs sont des fraudes. Cela
    rend le rappel observe verifiable.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        metrics: str = METRICS_TEXT,
        metrics_after: str | None = None,
    ) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []
        self._status = status
        self._metrics = metrics
        self._metrics_after = metrics_after
        self._scored = 0

    def post(self, url: str, json: dict, timeout: float = 0) -> FakeResponse:
        self.posts.append((url, json))
        if self._status != 200:
            return FakeResponse(None, self._status)
        if "transactions" in json:
            items = []
            for offset in range(len(json["transactions"])):
                flagged = (self._scored + offset) % 2 == 0
                items.append(
                    {
                        "index": offset,
                        "fraud_probability": 0.99 if flagged else 0.001,
                        "is_fraud": flagged,
                    }
                )
            self._scored += len(json["transactions"])
            return FakeResponse({"count": len(items), "predictions": items})
        flagged = self._scored % 2 == 0
        self._scored += 1
        return FakeResponse(
            {
                "fraud_probability": 0.99 if flagged else 0.001,
                "is_fraud": flagged,
                "threshold": 0.8,
                "threshold_source": "mlflow_run",
                "model_name": "fraud-detector",
                "model_version": "2",
            }
        )

    def get(self, url: str, timeout: float = 0) -> FakeResponse:
        """Le deuxieme releve peut differer : les compteurs Prometheus avancent."""
        self.gets.append(url)
        first = len(self.gets) == 1
        body = self._metrics if (first or not self._metrics_after) else self._metrics_after
        return FakeResponse(None, 200, body)

    def close(self) -> None:
        pass


class FakeClock:
    """Horloge et sommeil simules : l'ordonnancement se verifie sans attendre."""

    def __init__(self, step: float = 0.001) -> None:
        self.now = 1000.0
        self.step = step
        self.slept: list[float] = []

    def __call__(self) -> float:
        self.now += self.step
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make_stream(n: int = 10, frauds_every: int = 2) -> Stream:
    return Stream(
        payloads=[{name: float(i) for name in FEATURE_COLUMNS} for i in range(n)],
        labels=[1 if i % frauds_every == 0 else 0 for i in range(n)],
    )


# ===========================================================================
# Chargement : la propriete la plus importante
# ===========================================================================


@pytest.fixture
def processed_dir(tmp_path):
    frame = pd.DataFrame({name: [1.0, 2.0, 3.0, 4.0] for name in FEATURE_COLUMNS})
    frame[TARGET] = [0, 1, 0, 1]
    out = tmp_path / "processed"
    out.mkdir()
    frame.to_parquet(out / SPLIT_FILES["test"], index=False)
    return out


def test_class_is_never_part_of_a_payload(processed_dir) -> None:
    """LE test qui compte.

    Un vrai appelant ne possede pas l'etiquette. L'envoyer serait rejete par le
    contrat d'API (extra="forbid"), et si elle passait un jour, le service
    scorerait une variable qu'il n'a pas apprise.
    """
    stream = load_stream(processed_dir)

    for payload in stream.payloads:
        assert TARGET not in payload
    assert set(stream.payloads[0]) == set(FEATURE_COLUMNS)


def test_labels_are_kept_alongside(processed_dir) -> None:
    stream = load_stream(processed_dir)

    assert stream.labels == [0, 1, 0, 1]
    assert stream.n_frauds == 2
    assert len(stream) == 4


def test_class_never_reaches_the_service(processed_dir) -> None:
    """Verification de bout en bout : on inspecte ce qui a REELLEMENT ete poste."""
    session = RecordingSession()
    clock = FakeClock()
    stream = load_stream(processed_dir)

    run_campaign(
        stream,
        RunConfig(mode="batch", batch_size=2, warmup_requests=0),
        session=session,
        clock=clock,
        sleep=clock.sleep,
    )

    serialised = json.dumps(session.posts)
    assert TARGET not in serialised
    assert '"Class"' not in serialised


def test_missing_split_points_at_the_command(tmp_path) -> None:
    with pytest.raises(SimulatorError, match="python -m src.features"):
        load_stream(tmp_path / "vide")


# ===========================================================================
# Decoupage des requetes
# ===========================================================================


def test_batch_plan_covers_every_transaction_exactly_once() -> None:
    stream = make_stream(10)
    plan = build_requests(stream, RunConfig(mode="batch", batch_size=3))

    covered = [index for _, indices in plan for index in indices]
    assert covered == list(range(10))
    assert [len(indices) for _, indices in plan] == [3, 3, 3, 1]


def test_single_plan_sends_one_transaction_per_request() -> None:
    plan = build_requests(make_stream(5), RunConfig(mode="single"))

    assert len(plan) == 5
    assert all(len(indices) == 1 for _, indices in plan)
    assert all("transactions" not in body for body, _ in plan)


def test_batch_body_shape_matches_the_api_contract() -> None:
    plan = build_requests(make_stream(4), RunConfig(mode="batch", batch_size=2))

    body, _ = plan[0]
    assert set(body) == {"transactions"}
    assert len(body["transactions"]) == 2


def test_max_transactions_caps_the_plan() -> None:
    plan = build_requests(
        make_stream(100), RunConfig(mode="batch", batch_size=10, max_transactions=25)
    )

    assert sum(len(indices) for _, indices in plan) == 25


def test_stream_cycles_when_more_transactions_than_rows() -> None:
    """Un flux de production ne s'arrete pas parce qu'un fichier est epuise."""
    plan = build_requests(
        make_stream(4), RunConfig(mode="single", max_transactions=10)
    )

    covered = [index for _, indices in plan for index in indices]
    assert covered == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]


def test_empty_stream_is_rejected() -> None:
    with pytest.raises(SimulatorError, match="vide"):
        build_requests(Stream([], []), RunConfig())


def test_duration_with_a_rate_sizes_the_plan() -> None:
    config = RunConfig(mode="single", rate=50.0, duration=2.0)

    assert total_transactions(make_stream(1000), config) == 100


# ===========================================================================
# Ordonnancement en boucle ouverte
# ===========================================================================


def test_offsets_follow_the_target_rate() -> None:
    """100 tx/s en lots de 10 => une requete toutes les 0,1 s."""
    offsets = planned_offsets(count=4, rate=100.0, per_request=10)

    assert offsets == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_offsets_account_for_batch_size() -> None:
    """Le debit est en TRANSACTIONS/s : un lot plus gros s'espace davantage."""
    small = planned_offsets(3, rate=100.0, per_request=1)
    large = planned_offsets(3, rate=100.0, per_request=50)

    assert small == pytest.approx([0.0, 0.01, 0.02])
    assert large == pytest.approx([0.0, 0.5, 1.0])


def test_full_throttle_never_waits() -> None:
    assert planned_offsets(5, rate=0.0, per_request=1) == [0.0] * 5


def test_open_loop_schedules_on_the_clock_not_on_responses() -> None:
    """Le coeur de la parade contre l'omission coordonnee.

    Les instants prevus sont fixes d'avance. Une reponse lente ne repousse pas
    l'envoi suivant : elle produit du retard, qui est mesure.
    """
    session = RecordingSession()
    # step=0 : le temps n'avance QUE par les sommeils, donc l'horloge ne
    # devance pas l'ordonnancement et les attentes sont observables.
    clock = FakeClock(step=0.0)

    result = run_campaign(
        make_stream(6),
        RunConfig(mode="single", rate=1000.0, warmup_requests=0),
        session=session,
        clock=clock,
        sleep=clock.sleep,
    )

    assert [o.planned_offset for o in result.observations] == pytest.approx(
        [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]
    )
    assert clock.slept  # le simulateur a bien attendu pour tenir le rythme


def test_full_throttle_has_no_schedule_to_be_late_against() -> None:
    """A --rate 0 il n'y a aucun ordonnancement : tous les instants prevus
    valent zero. Mesurer la latence depuis "l'instant prevu" donnerait le temps
    ecoule CUMULE, ce qui ne signifie rien. La reference devient donc l'instant
    d'envoi, et le retard est nul par definition."""
    clock = FakeClock(step=0.01)

    result = run_campaign(
        make_stream(6),
        RunConfig(mode="single", rate=0.0, warmup_requests=0),
        session=RecordingSession(),
        clock=clock,
        sleep=clock.sleep,
    )

    for observation in result.observations:
        assert observation.latency_ms == pytest.approx(observation.service_ms)
        assert observation.scheduling_delay_ms == pytest.approx(0.0)
    assert not clock.slept


def test_report_says_the_open_loop_section_is_moot_at_full_throttle() -> None:
    """Afficher un retard d'ordonnancement sans ordonnancement serait trompeur."""
    clock = FakeClock()
    result = run_campaign(
        make_stream(4),
        RunConfig(mode="batch", batch_size=2, rate=0.0, warmup_requests=0),
        session=RecordingSession(),
        clock=clock,
        sleep=clock.sleep,
    )

    report = render_report(result)
    assert "sans objet a plein regime" in report
    assert "retard d'ordonnancement max" not in report


def test_latency_is_measured_from_the_planned_instant() -> None:
    """Deux latences, et leur ecart est le retard d'ordonnancement.

    Mesurer depuis l'envoi reel seulement masquerait la file d'attente -- c'est
    exactement ce que l'omission coordonnee fait disparaitre.
    """
    session = RecordingSession()
    clock = FakeClock(step=0.01)  # chaque tick coute 10 ms

    result = run_campaign(
        make_stream(4),
        RunConfig(mode="single", rate=1000.0, warmup_requests=0),
        session=session,
        clock=clock,
        sleep=clock.sleep,
    )

    for observation in result.observations:
        assert observation.latency_ms >= observation.service_ms
        assert observation.scheduling_delay_ms >= 0


# ===========================================================================
# Statistiques
# ===========================================================================


def test_percentile_uses_nearest_rank_without_interpolation() -> None:
    """Un p95 doit etre une valeur REELLEMENT observee."""
    values = list(range(1, 101))  # 1..100

    assert percentile(values, 0.50) == 50
    assert percentile(values, 0.95) == 95
    assert percentile(values, 0.99) == 99
    assert percentile(values, 1.0) == 100


def test_percentile_on_a_single_value() -> None:
    assert percentile([7.0], 0.99) == 7.0


def test_percentile_of_nothing_is_nan() -> None:
    assert math.isnan(percentile([], 0.5))


def test_latency_summary_on_a_known_sample() -> None:
    summary = LatencySummary.of([10.0, 20.0, 30.0, 40.0])

    assert summary.count == 4
    assert summary.p50 == 20.0
    assert summary.mean == 25.0
    assert summary.worst == 40.0


def test_summary_flags_an_unreliable_sample() -> None:
    """Ma propre mesure sur 3 lots avait donne un p95 absurde : le rapport doit
    le signaler plutot qu'afficher un nombre trompeur."""
    assert LatencySummary.of([1.0, 2.0, 3.0]).reliable is False
    assert LatencySummary.of([1.0] * MIN_SAMPLES_FOR_PERCENTILES).reliable is True


def test_histogram_quantile_interpolates_inside_the_bucket() -> None:
    """La mecanique de Prometheus : 10 observations, 2 sous 0,01 et 8 sous 0,02.

    Le p50 (rang 5) tombe dans la tranche ]0,01 ; 0,02], a 3/6 du chemin.
    """
    buckets = [(0.01, 2.0), (0.02, 8.0), (float("inf"), 10.0)]

    assert histogram_quantile(buckets, 0.50) == pytest.approx(0.015)
    assert histogram_quantile(buckets, 0.20) == pytest.approx(0.01)


def test_histogram_quantile_handles_the_infinite_bucket() -> None:
    buckets = [(0.01, 5.0), (float("inf"), 10.0)]

    assert histogram_quantile(buckets, 0.99) == pytest.approx(0.01)


def test_histogram_quantile_of_an_empty_histogram_is_nan() -> None:
    assert math.isnan(histogram_quantile([], 0.5))
    assert math.isnan(histogram_quantile([(0.1, 0.0)], 0.5))


# ===========================================================================
# Vue serveur
# ===========================================================================


def test_snapshot_reads_only_the_requested_endpoint() -> None:
    """Le texte d'exposition melange single et batch : on ne doit pas les
    confondre."""
    snapshot = read_server_snapshot(RecordingSession(), "http://x", "batch")

    assert snapshot.calls == 10.0
    assert snapshot.seconds == pytest.approx(0.15)
    assert snapshot.transactions == {"fraud": 4.0, "legitimate": 96.0}
    assert snapshot.mean_ms == pytest.approx(15.0)


def test_snapshot_difference_isolates_the_campaign() -> None:
    """Les compteurs Prometheus sont cumulatifs depuis le demarrage du service.
    Sans difference, on melangerait notre trafic aux appels faits depuis /docs."""
    before = ServerSnapshot(10.0, 0.10, {"fraud": 5.0}, [(0.01, 4.0), (float("inf"), 10.0)])
    after = ServerSnapshot(30.0, 0.40, {"fraud": 9.0}, [(0.01, 10.0), (float("inf"), 30.0)])

    delta = after.since(before)

    assert delta.calls == 20.0
    assert delta.seconds == pytest.approx(0.30)
    assert delta.transactions == {"fraud": 4.0}
    assert delta.buckets == [(0.01, 6.0), (float("inf"), 20.0)]
    assert delta.mean_ms == pytest.approx(15.0)


def test_campaign_reads_metrics_before_and_after() -> None:
    session = RecordingSession()
    clock = FakeClock()

    result = run_campaign(
        make_stream(4),
        RunConfig(mode="batch", batch_size=2, warmup_requests=0),
        session=session,
        clock=clock,
        sleep=clock.sleep,
    )

    assert len([url for url in session.gets if url.endswith("/metrics")]) == 2
    assert result.server is not None
    assert result.server.calls == 0.0  # les deux releves sont identiques ici


def test_unreachable_metrics_does_not_break_the_campaign() -> None:
    """On perd la colonne serveur du rapport, pas la mesure client."""

    class NoMetrics(RecordingSession):
        def get(self, url: str, timeout: float = 0) -> FakeResponse:
            raise requests.ConnectionError("pas de /metrics")

    clock = FakeClock()
    result = run_campaign(
        make_stream(4),
        RunConfig(mode="batch", batch_size=2, warmup_requests=0),
        session=NoMetrics(),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.server is None
    assert result.n_transactions == 4
    assert "n/a" in render_report(result)


# ===========================================================================
# Echauffement, echecs, rappel observe
# ===========================================================================


def test_warmup_requests_are_sent_but_excluded_from_measurements() -> None:
    session = RecordingSession()
    clock = FakeClock()

    result = run_campaign(
        make_stream(10),
        RunConfig(mode="single", warmup_requests=3),
        session=session,
        clock=clock,
        sleep=clock.sleep,
    )

    assert len(session.posts) == 13          # 3 d'echauffement + 10 mesurees
    assert len(result.observations) == 10    # mais seules 10 comptent


def test_failed_requests_are_counted_and_surfaced() -> None:
    clock = FakeClock()
    result = run_campaign(
        make_stream(4),
        RunConfig(mode="single", warmup_requests=0),
        session=RecordingSession(status=503),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.n_failures == 4
    assert "a investiguer" in render_report(result)


def test_network_errors_are_recorded_not_raised() -> None:
    """Une campagne de charge ne doit pas s'arreter a la premiere erreur reseau :
    le taux d'echec EST une mesure."""

    class Broken(RecordingSession):
        def post(self, url: str, json: dict, timeout: float = 0) -> FakeResponse:
            raise requests.ConnectionError("service tombe")

    clock = FakeClock()
    result = run_campaign(
        make_stream(3),
        RunConfig(mode="single", warmup_requests=0),
        session=Broken(),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.n_failures == 3
    assert all(o.status == 0 for o in result.observations)


def test_observed_recall_uses_the_kept_labels() -> None:
    """La doublure marque les indices pairs comme fraude, et le flux aussi :
    toutes les fraudes sont donc detectees, sans fausse alerte."""
    clock = FakeClock()
    result = run_campaign(
        make_stream(10, frauds_every=2),
        RunConfig(mode="batch", batch_size=10, warmup_requests=0),
        session=RecordingSession(),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.frauds_in_stream == 5
    assert result.detected == 5
    assert result.false_alerts == 0
    assert result.observed_recall == pytest.approx(1.0)


def test_report_marks_the_labels_as_simulation_only() -> None:
    """La production n'a pas les etiquettes : le rapport ne doit pas laisser
    croire le contraire."""
    clock = FakeClock()
    result = run_campaign(
        make_stream(4),
        RunConfig(mode="batch", batch_size=2, warmup_requests=0),
        session=RecordingSession(),
        clock=clock,
        sleep=clock.sleep,
    )

    report = render_report(result)
    assert "SIMULATION SEULEMENT" in report
    assert "rejets de paiement" in report


# ===========================================================================
# L'avertissement de contamination
# ===========================================================================


@pytest.mark.parametrize(
    "url", ["http://localhost:8000", "http://127.0.0.1:8000", "http://0.0.0.0:8000"]
)
def test_localhost_is_flagged_as_outside_docker(url: str) -> None:
    assert is_outside_docker(url) is True


@pytest.mark.parametrize("url", ["http://scorer:8000", "http://fraud-api.internal"])
def test_service_names_are_not_flagged(url: str) -> None:
    assert is_outside_docker(url) is False


def test_campaign_attaches_the_warning_when_tiring_from_outside() -> None:
    """Mesure : 54,81 ms depuis Windows contre 6,12 ms dans le reseau. Sans cet
    avertissement, on lirait des chiffres 9x pessimistes sans le savoir."""
    clock = FakeClock()
    result = run_campaign(
        make_stream(4),
        RunConfig(base_url="http://localhost:8000", mode="batch", batch_size=2, warmup_requests=0),
        session=RecordingSession(),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.warning == OUTSIDE_DOCKER_WARNING
    assert "Docker Desktop" in result.warning


def test_no_warning_from_inside_the_network() -> None:
    clock = FakeClock()
    result = run_campaign(
        make_stream(4),
        RunConfig(base_url="http://scorer:8000", mode="batch", batch_size=2, warmup_requests=0),
        session=RecordingSession(),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.warning is None


# ===========================================================================
# Configuration et rapport
# ===========================================================================


def test_config_routes_to_the_right_endpoint() -> None:
    single = RunConfig(base_url="http://x:8000", mode="single")
    batch = RunConfig(base_url="http://x:8000", mode="batch")

    assert single.url == "http://x:8000/predict"
    assert single.endpoint == "single"
    assert single.per_request == 1
    assert batch.url == "http://x:8000/predict/batch"
    assert batch.endpoint == "batch"
    assert batch.per_request == 500


def test_trailing_slash_does_not_double_up() -> None:
    assert RunConfig(base_url="http://x:8000/").url == "http://x:8000/predict/batch"


def test_report_shows_both_columns_and_their_gap() -> None:
    clock = FakeClock()
    result = run_campaign(
        make_stream(200),
        RunConfig(mode="batch", batch_size=2, warmup_requests=0),
        session=RecordingSession(metrics_after=METRICS_AFTER),
        clock=clock,
        sleep=clock.sleep,
    )

    report = render_report(result)
    assert "client" in report
    assert "serveur" in report
    assert "ecart p50" in report
    assert "BOUCLE OUVERTE" in report


# ===========================================================================
# Verification de /ready avant la campagne
# ===========================================================================


class ReadySession(RecordingSession):
    """Doublure qui repond a /ready avec un etat choisi."""

    def __init__(self, *, status: int = 200, body: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ready_status = status
        self._ready_body = body if body is not None else {
            "status": "ready",
            "reason": None,
            "model": {
                "model_name": "fraud-detector",
                "version": "2",
                "model_kind": "xgb",
                "threshold": 0.8080154657363892,
                "threshold_source": "mlflow_run",
            },
        }

    def get(self, url: str, timeout: float = 0) -> FakeResponse:
        self.gets.append(url)
        if url.endswith("/ready"):
            return FakeResponse(self._ready_body, self._ready_status)
        return super().get(url, timeout)


def test_readiness_check_targets_the_ready_endpoint() -> None:
    session = ReadySession()

    readiness = check_readiness(session, "http://scorer:8000/")

    assert readiness.ready is True
    assert session.gets == ["http://scorer:8000/ready"]


def test_ready_service_reports_the_served_version() -> None:
    readiness = check_readiness(ReadySession(), "http://scorer:8000")

    rendered = readiness.render("http://scorer:8000")
    assert "v2" in rendered
    assert "xgb" in rendered
    assert "mlflow_run" in rendered


def test_degraded_service_yields_an_actionable_message() -> None:
    """LE point de cette verification.

    Sans elle, un scorer degrade absorbe toute la campagne en 503 et produit un
    rapport rempli d'echecs ou la cause n'apparait nulle part.
    """
    session = ReadySession(
        status=503,
        body={
            "status": "not_ready",
            "reason": "Aucun modele sous l'alias @production pour 'fraud-detector'.",
            "model": None,
        },
    )

    readiness = check_readiness(session, "http://scorer:8000")

    assert readiness.ready is False
    assert readiness.status == 503
    rendered = readiness.render("http://scorer:8000")
    assert "n'est pas pret" in rendered
    assert "@production" in rendered
    assert "curl http://scorer:8000/ready" in rendered
    assert "--no-ready-check" in rendered


def test_unreachable_service_is_reported_not_raised() -> None:
    class Unreachable(ReadySession):
        def get(self, url: str, timeout: float = 0) -> FakeResponse:
            raise requests.ConnectionError("connection refused")

    readiness = check_readiness(Unreachable(), "http://scorer:8000")

    assert readiness.ready is False
    assert readiness.status == 0
    assert "injoignable" in (readiness.reason or "")


def test_non_json_response_does_not_crash_the_check() -> None:
    """Un intermediaire (proxy, load balancer) peut renvoyer du HTML."""

    class Html(ReadySession):
        def get(self, url: str, timeout: float = 0) -> FakeResponse:
            response = FakeResponse(None, 502)
            response.json = lambda: (_ for _ in ()).throw(ValueError("pas du JSON"))
            return response

    readiness = check_readiness(Html(), "http://scorer:8000")

    assert readiness.ready is False
    assert readiness.status == 502


def test_readiness_render_is_stable_without_a_model_block() -> None:
    readiness = Readiness(ready=False, status=503)

    rendered = readiness.render("http://x:8000")
    assert "n'est pas pret" in rendered
