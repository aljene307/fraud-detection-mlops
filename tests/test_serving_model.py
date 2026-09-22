"""Tests du chargeur de modele de production (step 2.A).

Aucun test ne charge le vrai modele, sauf celui marque ``requires_data`` :
l'operation coute ~13,8 s et exige une base MLflow que la CI n'aura pas. Les
doublures ci-dessous reproduisent la forme des objets MLflow, ce qui suffit a
verifier la logique -- qui est tout l'interet de ce module.
"""

from __future__ import annotations

from typing import Any

import pytest
from mlflow.exceptions import MlflowException

from src.config import MLFLOW_DB, PRODUCTION_ALIAS, REGISTERED_MODEL
from src.serving.model import (
    DEFAULT_TRACKING_WAIT_SECONDS,
    THRESHOLD_ENV_VAR,
    TRACKING_WAIT_ENV_VAR,
    ModelBundle,
    ModelLoadError,
    extract_feature_columns,
    load_production_model,
    resolve_production_version,
    resolve_threshold,
    tracking_wait_seconds,
    wait_for_tracking_server,
)

# Colonnes volontairement DIFFERENTES de FEATURE_COLUMNS : c'est ce qui permet
# de prouver que le chargeur lit la signature et non le code source.
SIGNATURE_COLUMNS = ("alpha", "beta", "gamma")

REAL_THRESHOLD = 0.8080154657363892


# ---------------------------------------------------------------------------
# Doublures
# ---------------------------------------------------------------------------


class FakeSchema:
    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names

    def input_names(self) -> list[str]:
        return list(self._names)


class FakeMetadata:
    def __init__(self, names: tuple[str, ...] | None) -> None:
        self._schema = FakeSchema(names) if names is not None else None

    def get_input_schema(self) -> FakeSchema | None:
        return self._schema


class FakeModel:
    def __init__(self, names: tuple[str, ...] | None = SIGNATURE_COLUMNS) -> None:
        self.metadata = FakeMetadata(names)


class FakeRunData:
    def __init__(self, metrics: dict, tags: dict, params: dict) -> None:
        self.metrics = metrics
        self.tags = tags
        self.params = params


class FakeRunInfo:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class FakeRun:
    def __init__(
        self,
        run_id: str = "run-abc",
        *,
        threshold: float | None = REAL_THRESHOLD,
        test_pr_auc: float | None = 0.8727,
    ) -> None:
        metrics: dict[str, float] = {}
        if threshold is not None:
            metrics["frozen_threshold"] = threshold
        if test_pr_auc is not None:
            metrics["test_pr_auc"] = test_pr_auc
        self.info = FakeRunInfo(run_id)
        self.data = FakeRunData(
            metrics=metrics,
            tags={
                "git_commit": "c3b4ef66a008647a72ff74094e92364dac5a0895",
                "git_dirty": "false",
                "dataset_sha256": "76274b69" + "0" * 56,
            },
            params={"model_kind": "xgb"},
        )


class FakeModelVersion:
    def __init__(self, version: str = "2", run_id: str = "run-abc") -> None:
        self.version = version
        self.run_id = run_id


class FakeClient:
    """MlflowClient reduit aux deux appels que le chargeur utilise."""

    def __init__(
        self,
        *,
        version: FakeModelVersion | None = None,
        run: FakeRun | None = None,
        alias_missing: bool = False,
    ) -> None:
        self._version = version or FakeModelVersion()
        self._run = run or FakeRun()
        self._alias_missing = alias_missing
        self.calls: list[str] = []

    def get_model_version_by_alias(self, name: str, alias: str) -> FakeModelVersion:
        self.calls.append(f"alias:{name}@{alias}")
        if self._alias_missing:
            raise MlflowException(f"Registered model alias {alias} not found.")
        return self._version

    def get_run(self, run_id: str) -> FakeRun:
        self.calls.append(f"run:{run_id}")
        return self._run


def _load(**kwargs: Any) -> ModelBundle:
    """Chargement avec doublures et un environnement vide par defaut."""
    kwargs.setdefault("client", FakeClient())
    kwargs.setdefault("loader", lambda uri: FakeModel())
    kwargs.setdefault("env", {})
    return load_production_model(**kwargs)


# ===========================================================================
# Resolution de l'alias
# ===========================================================================


def test_alias_is_resolved_to_a_version_and_its_run() -> None:
    client = FakeClient()

    bundle = _load(client=client)

    assert bundle.version == "2"
    assert bundle.run_id == "run-abc"
    assert client.calls == [
        f"alias:{REGISTERED_MODEL}@{PRODUCTION_ALIAS}",
        "run:run-abc",
    ]


def test_missing_alias_says_what_to_run() -> None:
    """Le message doit debloquer, pas seulement constater. C'est cette erreur
    que le step 2.B attrapera pour demarrer degrade."""
    client = FakeClient(alias_missing=True)

    with pytest.raises(ModelLoadError) as err:
        resolve_production_version(client)

    message = str(err.value)
    assert "python -m src.training.train --promote" in message
    assert f"@{PRODUCTION_ALIAS}" in message


def test_missing_alias_prevents_any_model_loading() -> None:
    """Inutile de payer 13,8 s de chargement si l'alias n'existe pas."""

    def exploding_loader(uri: str) -> Any:
        raise AssertionError("le modele n'aurait pas du etre charge")

    with pytest.raises(ModelLoadError):
        _load(client=FakeClient(alias_missing=True), loader=exploding_loader)


# ===========================================================================
# Le seuil : env -> run -> refus
# ===========================================================================


def test_threshold_comes_from_the_run_by_default() -> None:
    bundle = _load()

    assert bundle.threshold == pytest.approx(REAL_THRESHOLD)
    assert bundle.threshold_source == "mlflow_run"


def test_environment_variable_takes_priority() -> None:
    """Le levier metier : ajuster le point de fonctionnement sans reentrainer."""
    bundle = _load(env={THRESHOLD_ENV_VAR: "0.65"})

    assert bundle.threshold == pytest.approx(0.65)
    assert bundle.threshold_source == "env"


def test_threshold_source_is_always_visible() -> None:
    """Une variable oubliee d'un deploiement precedent s'appliquerait
    silencieusement. describe() la rend visible dans /ready et les logs."""
    described = _load(env={THRESHOLD_ENV_VAR: "0.65"}).describe()

    assert described["threshold_source"] == "env"
    assert described["threshold"] == pytest.approx(0.65)


def test_empty_environment_variable_falls_back_to_the_run() -> None:
    """FRAUD_THRESHOLD="" est frequent dans un docker-compose : ce n'est pas
    une valeur, c'est une absence."""
    bundle = _load(env={THRESHOLD_ENV_VAR: "   "})

    assert bundle.threshold_source == "mlflow_run"


def test_non_numeric_environment_variable_is_rejected() -> None:
    with pytest.raises(ModelLoadError, match="n'est pas un nombre"):
        _load(env={THRESHOLD_ENV_VAR: "haut"})


@pytest.mark.parametrize("value", ["80", "-0.1", "1.5"])
def test_environment_threshold_outside_zero_one_is_rejected(value: str) -> None:
    """80 au lieu de 0.80 : un seuil se compare a une probabilite."""
    with pytest.raises(ModelLoadError, match=r"\[0, 1\]"):
        _load(env={THRESHOLD_ENV_VAR: value})


def test_no_threshold_anywhere_is_a_refusal_not_a_default() -> None:
    """Le cas reel : logreg n'atteint jamais 90 % de precision, donc son run ne
    porte aucun frozen_threshold. Servir avec un seuil invente serait pire que
    ne pas servir."""
    run = FakeRun(threshold=None)

    with pytest.raises(ModelLoadError) as err:
        resolve_threshold(run, env={})

    message = str(err.value)
    assert "Refus de servir" in message
    assert THRESHOLD_ENV_VAR in message


def test_threshold_is_resolved_before_the_expensive_load() -> None:
    """L'ordre compte : echouer en microsecondes sur une configuration
    incomplete plutot qu'apres 13,8 s de chargement destine a etre jete."""

    def exploding_loader(uri: str) -> Any:
        raise AssertionError("le chargement ne doit pas etre tente sans seuil")

    with pytest.raises(ModelLoadError, match="Refus de servir"):
        _load(client=FakeClient(run=FakeRun(threshold=None)), loader=exploding_loader)


# ===========================================================================
# Les colonnes viennent de la signature, pas du code source
# ===========================================================================


def test_feature_columns_come_from_the_deployed_signature() -> None:
    """LE point de conception du module.

    La signature decrit ce sur quoi le modele DEPLOYE a ete entraine ;
    FEATURE_COLUMNS decrit ce que le code ACTUEL produit. Ajouter une variable
    au pipeline sans reentrainer ferait diverger les deux.
    """
    from src.features.build import FEATURE_COLUMNS

    bundle = _load()

    assert bundle.feature_columns == SIGNATURE_COLUMNS
    assert bundle.feature_columns != FEATURE_COLUMNS
    assert bundle.describe()["n_features"] == len(SIGNATURE_COLUMNS)


def test_model_without_signature_is_rejected() -> None:
    with pytest.raises(ModelLoadError, match="signature"):
        extract_feature_columns(FakeModel(names=None))


def test_model_with_empty_signature_is_rejected() -> None:
    with pytest.raises(ModelLoadError, match="aucune colonne"):
        extract_feature_columns(FakeModel(names=()))


def test_object_without_metadata_is_rejected() -> None:
    with pytest.raises(ModelLoadError, match="signature"):
        extract_feature_columns(object())


# ===========================================================================
# Le bundle
# ===========================================================================


def test_bundle_carries_provenance_for_auditing() -> None:
    """Une prediction doit pouvoir remonter au commit et aux octets qui ont
    produit le modele."""
    bundle = _load()

    assert bundle.git_commit == "c3b4ef66a008647a72ff74094e92364dac5a0895"
    assert bundle.git_dirty == "false"
    assert bundle.dataset_sha256.startswith("76274b69")
    assert bundle.test_pr_auc == pytest.approx(0.8727)
    assert bundle.model_kind == "xgb"


def test_bundle_tolerates_a_run_without_provenance_tags() -> None:
    """Une metadonnee manquante ne doit pas empecher de servir : elle devient
    'unknown', ce qui reste honnete."""
    run = FakeRun()
    run.data.tags = {}
    run.data.params = {}
    run.data.metrics.pop("test_pr_auc")

    bundle = _load(client=FakeClient(run=run))

    assert bundle.git_commit == "unknown"
    assert bundle.model_kind == "unknown"
    assert bundle.test_pr_auc is None


def test_describe_is_json_friendly() -> None:
    """describe() alimentera /ready : que des types serialisables, et surtout
    pas l'objet modele lui-meme."""
    import json

    described = _load().describe()

    json.dumps(described)  # ne doit pas lever
    assert "model" not in described
    assert described["version"] == "2"


def test_bundle_repr_does_not_dump_the_model() -> None:
    """field(repr=False) : afficher un bundle dans un log ne doit pas deverser
    un booster entier."""
    assert "FakeModel" not in repr(_load())


def test_loader_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_loader(uri: str) -> Any:
        raise OSError("magasin d'artefacts injoignable")

    with pytest.raises(ModelLoadError, match="magasin d'artefacts injoignable"):
        _load(loader=failing_loader)


# ===========================================================================
# Vrai modele : local seulement, saute en CI
# ===========================================================================


@pytest.mark.requires_data
@pytest.mark.skipif(not MLFLOW_DB.exists(), reason="mlflow.db absent (normal en CI)")
def test_real_production_model_loads_with_its_threshold() -> None:
    """Le seul test qui paie les ~13,8 s de chargement reel.

    Il verifie que la chaine complete alias -> version -> run -> seuil tient sur
    la vraie base, et que la signature declare bien 30 colonnes.
    """
    bundle = load_production_model(env={})

    assert bundle.threshold_source == "mlflow_run"
    assert 0.0 < bundle.threshold < 1.0
    assert len(bundle.feature_columns) == 30
    assert "log_amount" in bundle.feature_columns
    assert "hour_of_day" in bundle.feature_columns
    assert "Time" not in bundle.feature_columns
    assert bundle.model_kind == "xgb"


# ===========================================================================
# Attente du serveur de tracking
# ===========================================================================


class FakeWaitClock:
    """Horloge et sommeil simules : l'attente se verifie sans attendre."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_local_store_needs_no_waiting() -> None:
    """Un store sqlite n'a pas de serveur a attendre."""
    clock = FakeWaitClock()

    assert wait_for_tracking_server(
        "sqlite:///mlflow.db", clock=clock, sleep=clock.sleep, probe=lambda _: False
    ) is True
    assert clock.slept == []


def test_reachable_server_returns_immediately() -> None:
    clock = FakeWaitClock()
    calls: list[str] = []

    assert wait_for_tracking_server(
        "http://mlflow:5000",
        clock=clock,
        sleep=clock.sleep,
        probe=lambda url: calls.append(url) or True,
    ) is True
    assert clock.slept == []
    assert calls == ["http://mlflow:5000/health"]


def test_server_that_comes_up_late_is_waited_for() -> None:
    """LE scenario observe : au redemarrage du demon Docker, les conteneurs
    remontent simultanement et MLflow n'ecoute pas encore.

    depends_on n'ordonne que `docker compose up`, donc la parade doit vivre
    dans l'application.
    """
    clock = FakeWaitClock()
    attempts = {"n": 0}

    def probe(url: str) -> bool:
        attempts["n"] += 1
        return attempts["n"] >= 4          # repond a la 4e tentative

    assert wait_for_tracking_server(
        "http://mlflow:5000",
        deadline_seconds=90.0,
        interval_seconds=2.0,
        clock=clock,
        sleep=clock.sleep,
        probe=probe,
    ) is True
    assert clock.slept == [2.0, 2.0, 2.0]
    assert attempts["n"] == 4


def test_waiting_gives_up_at_the_deadline() -> None:
    """L'attente est BORNEE : sinon le port resterait ferme indefiniment."""
    clock = FakeWaitClock()

    assert wait_for_tracking_server(
        "http://mlflow:5000",
        deadline_seconds=10.0,
        interval_seconds=2.0,
        clock=clock,
        sleep=clock.sleep,
        probe=lambda _: False,
    ) is False
    assert sum(clock.slept) <= 10.0


def test_probe_targets_the_health_endpoint() -> None:
    """/health est EXEMPTE de la validation d'hote de MLflow 3 : il repond meme
    avant qu'on ait regle MLFLOW_SERVER_ALLOWED_HOSTS."""
    seen: list[str] = []
    clock = FakeWaitClock()

    wait_for_tracking_server(
        "http://mlflow:5000/",
        clock=clock,
        sleep=clock.sleep,
        probe=lambda url: seen.append(url) or True,
    )

    assert seen == ["http://mlflow:5000/health"]


def test_wait_deadline_is_configurable() -> None:
    assert tracking_wait_seconds({}) == DEFAULT_TRACKING_WAIT_SECONDS
    assert tracking_wait_seconds({TRACKING_WAIT_ENV_VAR: "15"}) == 15.0
    assert tracking_wait_seconds({TRACKING_WAIT_ENV_VAR: "   "}) == DEFAULT_TRACKING_WAIT_SECONDS
    # Une valeur illisible ne doit pas empecher le service de demarrer.
    assert tracking_wait_seconds({TRACKING_WAIT_ENV_VAR: "beaucoup"}) == DEFAULT_TRACKING_WAIT_SECONDS
