"""Rejoue le test set en flux vers l'API et mesure debit et latences.

Point d'entree : ``python -m src.simulator`` (voir __main__.py).

Trois principes gouvernent ce module, chacun issu d'une mesure faite sur la pile
reelle avant d'ecrire une ligne.

**1. On mesure d'ou l'on tire.** Depuis Windows contre localhost:8000, un
POST /predict prend 54,81 ms ; depuis l'interieur du reseau Docker, 6,12 ms. Les
~48 ms d'ecart sont le port forwarding de Docker Desktop, et ils ne touchent que
les POST -- un GET /health reste a 1,3 ms des deux cotes, ce qui rend le piege
sournois. Le simulateur avertit donc quand il tire depuis l'exterieur.

**2. Boucle OUVERTE, pas fermee.** Attendre la reponse avant de planifier
l'envoi suivant ferait qu'un serveur ralenti recevrait mecaniquement moins de
trafic : la file d'attente qu'un vrai flux subirait n'apparaitrait jamais et les
percentiles hauts seraient flatteurs. C'est l'omission coordonnee. Les envois
sont donc planifies sur l'horloge, et la latence mesuree depuis l'instant PREVU
autant que depuis l'instant reel -- leur ecart est le retard d'ordonnancement,
qui est lui-meme un signal.

**3. Deux nombres, pas un.** Mesure : 6,12 ms pour une transaction seule, 0,08 ms
par transaction dans un lot de 200 -- 76 fois moins, parce que hors modele le
travail est FIXE PAR REQUETE (~0,2 ms) et non par transaction. Une vraie API de
fraude etant appelee une transaction a la fois, le debit par lot n'est PAS ce que
verrait la production : /predict donne la latence d'un appelant reel,
/predict/batch le plafond de debit du modele. Les deux sont legitimes.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from prometheus_client.parser import text_string_to_metric_families

from src.config import DATA_PROCESSED, TARGET
from src.features.build import SPLIT_FILES

DEFAULT_BASE_URL = "http://localhost:8000"

# Hotes qui trahissent un tir depuis l'exterieur du reseau Docker.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

# En dessous de ce nombre de mesures, un p99 est un point, pas un percentile.
MIN_SAMPLES_FOR_PERCENTILES = 100

OUTSIDE_DOCKER_WARNING = (
    "ATTENTION : tir depuis l'exterieur du reseau Docker.\n"
    "  Mesure faite sur cette pile : POST /predict = 54,81 ms depuis Windows\n"
    "  contre 6,12 ms depuis le reseau interne. Les ~48 ms d'ecart sont le port\n"
    "  forwarding de Docker Desktop, pas le service -- et ils ne touchent que les\n"
    "  POST, un GET /health restant a 1,3 ms des deux cotes.\n"
    "  Pour mesurer le SERVICE, lance le simulateur dans le reseau."
)


class SimulatorError(RuntimeError):
    """Campagne impossible : donnees absentes ou incoherentes."""


def is_outside_docker(base_url: str) -> bool:
    """Vrai si l'URL vise localhost, donc probablement hors du reseau Docker."""
    return (urlparse(base_url).hostname or "") in LOCAL_HOSTS


# ---------------------------------------------------------------------------
# Le flux
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stream:
    """Les charges utiles a envoyer, et les etiquettes gardees A PART."""

    payloads: list[dict[str, float]]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.payloads)

    @property
    def n_frauds(self) -> int:
        return sum(self.labels)


def load_stream(processed_dir: Path = DATA_PROCESSED, split: str = "test") -> Stream:
    """Charge un split en SEPARANT les etiquettes des charges utiles.

    ``Class`` ne doit jamais partir dans une requete : un vrai appelant ne la
    possede pas, et le contrat d'API la rejetterait (extra="forbid"). On la garde
    neanmoins de cote pour rapporter le rappel observe -- une capacite de
    SIMULATION que la production n'a pas, les vraies etiquettes de fraude
    n'arrivant que des semaines plus tard, par les rejets de paiement.
    """
    path = processed_dir / SPLIT_FILES[split]
    if not path.exists():
        raise SimulatorError(
            f"{path} est introuvable.\n"
            "  Lance d'abord le decoupage :  python -m src.features"
        )

    frame = pd.read_parquet(path)
    if TARGET not in frame.columns:
        raise SimulatorError(f"{path.name} ne contient pas la colonne {TARGET!r}.")

    return Stream(
        payloads=frame.drop(columns=[TARGET]).to_dict("records"),
        labels=frame[TARGET].astype(int).tolist(),
    )


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------


def percentile(ordered: Sequence[float], share: float) -> float:
    """Percentile par rang le plus proche, sans interpolation.

    Convention des outils de charge : un p95 doit etre une valeur REELLEMENT
    observee. Interpoler inventerait une latence que personne n'a subie.
    """
    if not ordered:
        return math.nan
    index = math.ceil(share * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


@dataclass(frozen=True)
class LatencySummary:
    count: int
    p50: float
    p95: float
    p99: float
    mean: float
    worst: float

    @classmethod
    def of(cls, values_ms: Sequence[float]) -> LatencySummary:
        if not values_ms:
            return cls(0, math.nan, math.nan, math.nan, math.nan, math.nan)
        ordered = sorted(values_ms)
        return cls(
            count=len(ordered),
            p50=percentile(ordered, 0.50),
            p95=percentile(ordered, 0.95),
            p99=percentile(ordered, 0.99),
            mean=sum(ordered) / len(ordered),
            worst=ordered[-1],
        )

    @property
    def reliable(self) -> bool:
        """Assez de mesures pour que les percentiles hauts signifient quelque chose."""
        return self.count >= MIN_SAMPLES_FOR_PERCENTILES


def histogram_quantile(buckets: Sequence[tuple[float, float]], share: float) -> float:
    """Quantile estime depuis des tranches cumulatives, comme le fait Prometheus.

    ``buckets`` est une liste (borne_superieure, compte_cumule) triee. On cherche
    la tranche contenant le rang voulu, puis on interpole lineairement dedans.

    C'est cette operation qui rend un Histogram superieur a un Summary : elle
    s'applique aussi bien a la SOMME des tranches de plusieurs pods, ce qu'une
    moyenne de percentiles ne permettrait pas.
    """
    if not buckets:
        return math.nan
    total = buckets[-1][1]
    if total <= 0:
        return math.nan

    rank = share * total
    lower_bound = 0.0
    lower_count = 0.0
    for upper, cumulative in buckets:
        if cumulative >= rank:
            if math.isinf(upper):
                return lower_bound
            span = cumulative - lower_count
            if span <= 0:
                return upper
            return lower_bound + (upper - lower_bound) * (rank - lower_count) / span
        lower_bound, lower_count = upper, cumulative
    return buckets[-1][0]


# ---------------------------------------------------------------------------
# Vue serveur, lue dans /metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServerSnapshot:
    """Etat des compteurs du service pour un endpoint, a un instant donne."""

    calls: float
    seconds: float
    transactions: dict[str, float]
    buckets: list[tuple[float, float]]

    def since(self, earlier: ServerSnapshot) -> ServerSnapshot:
        """Ecart entre deux releves : ce que la campagne seule a produit.

        Les compteurs Prometheus sont cumulatifs depuis le demarrage du service.
        Sans cette difference, on melangerait notre trafic a tout ce qui a
        precede -- y compris les appels faits depuis /docs.
        """
        previous = dict(earlier.buckets)
        return ServerSnapshot(
            calls=self.calls - earlier.calls,
            seconds=self.seconds - earlier.seconds,
            transactions={
                key: value - earlier.transactions.get(key, 0.0)
                for key, value in self.transactions.items()
            },
            buckets=[
                (upper, count - previous.get(upper, 0.0)) for upper, count in self.buckets
            ],
        )

    @property
    def mean_ms(self) -> float:
        return (self.seconds / self.calls * 1000) if self.calls else math.nan

    def quantile_ms(self, share: float) -> float:
        return histogram_quantile(self.buckets, share) * 1000


def read_server_snapshot(
    session: requests.Session, base_url: str, endpoint: str
) -> ServerSnapshot:
    """Lit /metrics et en extrait la vue serveur pour un endpoint."""
    response = session.get(f"{base_url.rstrip('/')}/metrics", timeout=10)
    response.raise_for_status()

    calls = seconds = 0.0
    transactions: dict[str, float] = {}
    buckets: list[tuple[float, float]] = []

    for family in text_string_to_metric_families(response.text):
        for item in family.samples:
            if item.labels.get("endpoint") != endpoint:
                continue
            if item.name == "fraud_prediction_latency_seconds_count":
                calls = item.value
            elif item.name == "fraud_prediction_latency_seconds_sum":
                seconds = item.value
            elif item.name == "fraud_prediction_latency_seconds_bucket":
                buckets.append((float(item.labels["le"]), item.value))
            elif item.name == "fraud_predictions_total":
                transactions[item.labels.get("decision", "?")] = item.value

    buckets.sort(key=lambda pair: pair[0])
    return ServerSnapshot(calls, seconds, transactions, buckets)


# ---------------------------------------------------------------------------
# Configuration et observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    base_url: str = DEFAULT_BASE_URL
    mode: str = "batch"
    rate: float = 0.0              # transactions/s ; 0 = plein regime
    batch_size: int = 500
    max_transactions: int | None = None
    duration: float | None = None
    warmup_requests: int = 3
    timeout: float = 30.0

    @property
    def endpoint(self) -> str:
        """Le nom d'etiquette utilise par le service dans ses metriques."""
        return "single" if self.mode == "single" else "batch"

    @property
    def url(self) -> str:
        suffix = "/predict" if self.mode == "single" else "/predict/batch"
        return f"{self.base_url.rstrip('/')}{suffix}"

    @property
    def per_request(self) -> int:
        return 1 if self.mode == "single" else max(1, self.batch_size)


@dataclass
class Observation:
    """Une requete envoyee, vue du client."""

    planned_offset: float
    latency_ms: float        # depuis l'instant PREVU : inclut le retard
    service_ms: float        # depuis l'instant d'envoi REEL
    n_transactions: int
    status: int
    scores: list[float] = field(default_factory=list)
    decisions: list[bool] = field(default_factory=list)

    @property
    def scheduling_delay_ms(self) -> float:
        """Ce que la boucle ouverte revele : le retard sur l'horloge."""
        return self.latency_ms - self.service_ms


def total_transactions(stream: Stream, config: RunConfig) -> int:
    """Combien de transactions la campagne doit envoyer.

    Avec une duree ET un debit cible, on prevoit de quoi tenir la duree, en
    reparcourant le flux si necessaire -- un flux de production ne s'arrete pas
    parce qu'on a epuise un fichier.
    """
    if config.max_transactions is not None:
        return max(1, config.max_transactions)
    if config.duration is not None and config.rate > 0:
        return max(1, math.ceil(config.duration * config.rate))
    return len(stream)


def build_requests(stream: Stream, config: RunConfig) -> list[tuple[dict[str, Any], list[int]]]:
    """Decoupe le flux en corps de requetes, avec les indices couverts.

    Les indices, pris modulo la taille du flux, servent a confronter les
    decisions du service aux etiquettes conservees -- sans jamais les envoyer.
    """
    if not stream.payloads:
        raise SimulatorError("Le flux est vide.")

    total = total_transactions(stream, config)
    size = config.per_request
    length = len(stream)

    plan: list[tuple[dict[str, Any], list[int]]] = []
    for start in range(0, total, size):
        indices = [position % length for position in range(start, min(start + size, total))]
        if config.mode == "single":
            plan.append((dict(stream.payloads[indices[0]]), indices))
        else:
            plan.append(
                ({"transactions": [stream.payloads[i] for i in indices]}, indices)
            )
    return plan


def planned_offsets(count: int, rate: float, per_request: int) -> list[float]:
    """Instants prevus, en secondes depuis le depart de la campagne.

    ``rate`` est un debit en TRANSACTIONS par seconde ; une requete en portant
    ``per_request``, l'intervalle entre requetes vaut per_request / rate. Un rate
    nul signifie plein regime : tous les instants prevus valent zero, donc on
    n'attend jamais.
    """
    if rate <= 0:
        return [0.0] * count
    interval = per_request / rate
    return [index * interval for index in range(count)]


def extract_predictions(body: Any, mode: str) -> tuple[list[float], list[bool]]:
    if mode == "single":
        return [float(body["fraud_probability"])], [bool(body["is_fraud"])]
    items = body.get("predictions", [])
    return (
        [float(item["fraud_probability"]) for item in items],
        [bool(item["is_fraud"]) for item in items],
    )


# ---------------------------------------------------------------------------
# Resultat
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    config: RunConfig
    observations: list[Observation]
    wall_seconds: float
    server: ServerSnapshot | None
    frauds_in_stream: int
    detected: int
    false_alerts: int
    warning: str | None = None

    @property
    def n_transactions(self) -> int:
        return sum(item.n_transactions for item in self.observations)

    @property
    def n_failures(self) -> int:
        return sum(1 for item in self.observations if item.status != 200)

    @property
    def throughput(self) -> float:
        return self.n_transactions / self.wall_seconds if self.wall_seconds > 0 else math.nan

    @property
    def client(self) -> LatencySummary:
        """Latence depuis l'instant PREVU : la mesure honnete en boucle ouverte."""
        return LatencySummary.of([item.latency_ms for item in self.observations])

    @property
    def service(self) -> LatencySummary:
        """Latence depuis l'envoi reel : le temps de service, hors retard."""
        return LatencySummary.of([item.service_ms for item in self.observations])

    @property
    def worst_scheduling_delay_ms(self) -> float:
        delays = [item.scheduling_delay_ms for item in self.observations]
        return max(delays) if delays else math.nan

    @property
    def last_score(self) -> float:
        for item in reversed(self.observations):
            if item.scores:
                return item.scores[-1]
        return math.nan

    @property
    def observed_recall(self) -> float:
        return self.detected / self.frauds_in_stream if self.frauds_in_stream else math.nan

    @property
    def observed_precision(self) -> float:
        flagged = self.detected + self.false_alerts
        return self.detected / flagged if flagged else math.nan


# ---------------------------------------------------------------------------
# La campagne
# ---------------------------------------------------------------------------


def run_campaign(
    stream: Stream,
    config: RunConfig,
    *,
    session: requests.Session | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[RunResult], None] | None = None,
) -> RunResult:
    """Rejoue le flux et renvoie les mesures des deux cotes.

    ``clock`` et ``sleep`` sont injectables : l'ordonnancement en boucle ouverte
    devient alors testable sans attendre reellement.

    ``time.perf_counter`` et non ``time.time`` : monotone, donc insensible aux
    ajustements NTP qui pourraient faire reculer l'horloge en pleine campagne.
    """
    owns_session = session is None
    active = session if session is not None else requests.Session()
    warning = OUTSIDE_DOCKER_WARNING if is_outside_docker(config.base_url) else None

    try:
        plan = build_requests(stream, config)

        # Echauffement : les premieres requetes paient des chemins froids
        # (imports paresseux, caches vides). On les envoie mais on les EXCLUT des
        # mesures, sinon elles polluent les percentiles hauts.
        for body, _ in plan[: config.warmup_requests]:
            try:
                active.post(config.url, json=body, timeout=config.timeout)
            except requests.RequestException:
                pass

        before = _safe_snapshot(active, config)
        offsets = planned_offsets(len(plan), config.rate, config.per_request)

        observations: list[Observation] = []
        detected = false_alerts = frauds_seen = 0
        # A plein regime il n'y a PAS d'ordonnancement : tous les instants prevus
        # valent zero. Mesurer la latence depuis "l'instant prevu" donnerait
        # alors le temps ecoule cumule, ce qui ne signifie rien. On prend donc
        # l'instant d'envoi comme reference, et le retard est nul par definition.
        scheduled = config.rate > 0
        started = clock()
        last_report = started

        for (body, indices), offset in zip(plan, offsets, strict=True):
            target = started + offset
            if scheduled:
                now = clock()
                if now < target:
                    sleep(target - now)

            sent = clock()
            reference = target if scheduled else sent
            try:
                response = active.post(config.url, json=body, timeout=config.timeout)
                status = response.status_code
                scores, decisions = (
                    extract_predictions(response.json(), config.mode)
                    if status == 200
                    else ([], [])
                )
            except requests.RequestException:
                status, scores, decisions = 0, [], []
            finished = clock()

            for position, flagged in zip(indices, decisions, strict=False):
                truth = stream.labels[position]
                frauds_seen += truth
                if flagged and truth:
                    detected += 1
                elif flagged and not truth:
                    false_alerts += 1

            observations.append(
                Observation(
                    planned_offset=offset,
                    latency_ms=(finished - reference) * 1000,
                    service_ms=(finished - sent) * 1000,
                    n_transactions=len(indices),
                    status=status,
                    scores=scores,
                    decisions=decisions,
                )
            )

            elapsed = finished - started
            if on_progress is not None and (finished - last_report) >= 1.0:
                last_report = finished
                on_progress(
                    RunResult(
                        config, observations, elapsed, None,
                        frauds_seen, detected, false_alerts, warning,
                    )
                )
            if config.duration is not None and elapsed >= config.duration:
                break

        wall = clock() - started
        after = _safe_snapshot(active, config)

        return RunResult(
            config=config,
            observations=observations,
            wall_seconds=wall,
            server=after.since(before) if (before and after) else None,
            frauds_in_stream=frauds_seen,
            detected=detected,
            false_alerts=false_alerts,
            warning=warning,
        )
    finally:
        if owns_session:
            active.close()


def _safe_snapshot(
    session: requests.Session, config: RunConfig
) -> ServerSnapshot | None:
    """Un /metrics injoignable ne doit pas faire echouer la campagne.

    On perd la colonne serveur du rapport, pas la mesure client.
    """
    try:
        return read_server_snapshot(session, config.base_url, config.endpoint)
    except (requests.RequestException, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Restitution
# ---------------------------------------------------------------------------


def _ratio(value: float) -> str:
    """Un ratio indefini s'ecrit 'n/a', pas 'nan%'.

    Le cas arrive des qu'aucune fraude n'a traverse le flux -- possible sur un
    echantillon court, puisqu'il n'y en a que 0,17 %.
    """
    return "n/a" if math.isnan(value) else f"{value:.1%}"


def render_progress(result: RunResult) -> str:
    """Une ligne, affichee environ chaque seconde pendant la campagne."""
    service = result.service
    return (
        f"[{result.wall_seconds:6.1f}s] "
        f"{result.throughput:9.1f} tx/s | "
        f"p50 {service.p50:6.2f} ms | "
        f"dernier score {result.last_score:.4f} | "
        f"fraudes {result.detected}/{result.frauds_in_stream}"
    )


def render_report(result: RunResult) -> str:
    config = result.config
    client, service = result.client, result.service
    bar = "=" * 72
    lot = f", lots de {config.batch_size}" if config.mode == "batch" else ""
    rate = "plein regime" if config.rate <= 0 else f"cible {config.rate:.0f} tx/s"

    lines = [
        "",
        bar,
        f"  CAMPAGNE TERMINEE  (mode {config.mode}{lot}, {rate})",
        bar,
        f"  transactions        {result.n_transactions:>12,}".replace(",", " "),
        f"  requetes            {len(result.observations):>12,}".replace(",", " "),
        f"  duree               {result.wall_seconds:>12.2f} s",
        f"  debit               {result.throughput:>12.1f} tx/s",
    ]
    if result.n_failures:
        lines.append(f"  requetes en echec   {result.n_failures:>12}  <- a investiguer")

    lines += [
        "",
        f"  LATENCE PAR REQUETE {'client':>14}{'serveur':>14}",
        "  " + "-" * 44,
    ]
    server = result.server
    for label, share, attr in (("p50", 0.50, "p50"), ("p95", 0.95, "p95"), ("p99", 0.99, "p99")):
        srv = f"{server.quantile_ms(share):>12.2f} ms" if server else f"{'n/a':>15}"
        lines.append(f"    {label:<16}{getattr(service, attr):>12.2f} ms{srv}")
    if server and not math.isnan(server.mean_ms):
        lines += [
            "  " + "-" * 44,
            f"    moyenne         {service.mean:>12.2f} ms{server.mean_ms:>12.2f} ms",
            (
                f"    ecart p50       {service.p50 - server.quantile_ms(0.50):>12.2f} ms"
                "   <- transport + validation + serialisation"
            ),
        ]

    if config.rate > 0:
        lines += [
            "",
            "  BOUCLE OUVERTE",
            (
                f"    latence depuis l'instant PREVU : p50 {client.p50:.2f} ms, "
                f"p99 {client.p99:.2f} ms"
            ),
            f"    retard d'ordonnancement max    : {result.worst_scheduling_delay_ms:.2f} ms",
        ]
        if result.worst_scheduling_delay_ms > 50:
            lines += [
                "    Le simulateur n'a pas tenu le rythme : soit le debit cible",
                "    depasse ce que le service absorbe, soit le simulateur manque",
                "    de CPU. Les percentiles client integrent cette attente.",
            ]
    else:
        lines += [
            "",
            "  BOUCLE OUVERTE : sans objet a plein regime (--rate 0)",
            "    Il n'y a aucun ordonnancement a respecter, donc pas de retard a",
            "    mesurer. Pour eprouver la tenue sous un debit donne, fixe --rate.",
        ]

    if not service.reliable:
        lines += [
            "",
            (
                f"  ATTENTION : {service.count} mesures seulement. En dessous de "
                f"{MIN_SAMPLES_FOR_PERCENTILES},"
            ),
            "  un p99 est un point isole, pas un percentile.",
        ]

    lines += [
        "",
        "  SIMULATION SEULEMENT -- la production n'a pas les etiquettes",
        "  (les vraies fraudes ne sont connues que des semaines plus tard, par",
        "   les rejets de paiement)",
        f"    fraudes dans le flux : {result.frauds_in_stream:>6}",
        (
            f"    detectees            : {result.detected:>6}"
            f"   (rappel observe {_ratio(result.observed_recall)})"
        ),
        (
            f"    fausses alertes      : {result.false_alerts:>6}"
            f"   (precision observee {_ratio(result.observed_precision)})"
        ),
        bar,
        "",
    ]
    return "\n".join(lines)
