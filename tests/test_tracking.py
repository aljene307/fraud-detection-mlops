"""Tests de l'enregistrement MLflow (step 1.C).

Tout se passe dans une base SQLite temporaire : la suite ne touche jamais au
mlflow.db du projet, et n'exige aucun serveur.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src.config import PRODUCTION_ALIAS, REGISTERED_MODEL, TARGET
from src.features.build import FEATURE_COLUMNS, SPLIT_FILES
from src.training import tracking
from src.training.metrics import confusion_at_threshold
from src.training.tracking import (
    TrackedRun,
    TrackingError,
    assert_no_forbidden_metrics,
    dataset_tags,
    git_tags,
    plot_confusion,
    plot_pr_curves,
    prefixed,
    promote_best,
)
from src.training.train import RunResult, load_splits, run_one

SPLIT_SPEC = {"train": (600, 40), "val": (200, 15), "test": (200, 15)}


def _make_split(n_rows: int, n_positives: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({column: rng.normal(size=n_rows) for column in FEATURE_COLUMNS})
    frame[TARGET] = 0
    positions = rng.choice(n_rows, size=n_positives, replace=False)
    frame.iloc[positions, frame.columns.get_loc(TARGET)] = 1
    frame.iloc[positions, frame.columns.get_loc("V1")] += 3.5
    frame.iloc[positions, frame.columns.get_loc("V2")] -= 3.0
    return frame


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    out = tmp_path / "processed"
    out.mkdir()
    for index, (name, (rows, positives)) in enumerate(SPLIT_SPEC.items()):
        _make_split(rows, positives, seed=index).to_parquet(
            out / SPLIT_FILES[name], index=False
        )
    (out / "split_manifest.json").write_text(
        json.dumps(
            {
                "strategy": "stratified",
                "seed": 42,
                "source": {"sha256": "b" * 64, "rows": 1000, "frauds": 70},
                "splits": {},
            }
        ),
        encoding="utf-8",
    )
    return out


@pytest.fixture
def splits(processed_dir: Path) -> dict:
    return load_splits(processed_dir)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Session MLflow isolee sur une base SQLite jetable."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.chdir(tmp_path)  # les artefacts atterrissent dans tmp_path
    return tracking.start_session(experiment="tests", tracking_uri=uri)


# ===========================================================================
# Garde-fou anti-accuracy
# ===========================================================================


def test_forbidden_metrics_are_rejected() -> None:
    """La regle de CLAUDE.md appliquee a l'execution.

    C'est aussi la raison pour laquelle mlflow.autolog() est proscrit : il
    journalise training_accuracy_score sans rien demander.
    """
    with pytest.raises(TrackingError, match="accuracy"):
        assert_no_forbidden_metrics({"test_pr_auc": 0.87, "test_accuracy": 0.999})


def test_legitimate_metrics_pass_the_guard() -> None:
    assert_no_forbidden_metrics({"val_pr_auc": 0.87, "test_recall_at_p90": 0.8})


def test_prefixing_separates_validation_from_test() -> None:
    assert prefixed({"pr_auc": 0.8}, "val") == {"val_pr_auc": 0.8}
    assert prefixed({"pr_auc": 0.8}, "test") == {"test_pr_auc": 0.8}


# ===========================================================================
# Tags de tracabilite
# ===========================================================================


def test_git_tags_report_commit_and_dirtiness() -> None:
    """Un run lance depuis un arbre modifie n'est pas rejouable a partir du
    seul SHA : il faut que ca se voie."""
    tags = git_tags()

    assert set(tags) == {"git_commit", "git_branch", "git_dirty"}
    assert tags["git_dirty"] in {"true", "false", "unknown"}
    if tags["git_commit"] != "unknown":
        assert len(tags["git_commit"]) == 40


def test_dataset_tags_read_the_split_manifest(processed_dir: Path) -> None:
    tags = dataset_tags(processed_dir)

    assert tags["dataset_sha256"] == "b" * 64
    assert tags["split_strategy"] == "stratified"


def test_dataset_tags_degrade_gracefully_without_manifest(tmp_path: Path) -> None:
    """Absence de manifeste : on tague 'unknown' plutot que de faire echouer
    tout l'entrainement pour une metadonnee."""
    assert dataset_tags(tmp_path / "vide")["dataset_sha256"] == "unknown"


# ===========================================================================
# Artefacts graphiques
# ===========================================================================


def test_pr_curve_plot_is_written(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    scores = rng.random(500)

    path = plot_pr_curves(
        {"validation": (y, scores), "test": (y, scores)},
        baseline=float(y.mean()),
        title="essai",
        path=tmp_path / "pr.png",
    )

    assert path.exists()
    assert path.stat().st_size > 1_000


def test_confusion_plot_is_written(tmp_path: Path) -> None:
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.full(90, 0.1), np.full(10, 0.9)])
    counts = confusion_at_threshold(y, scores, 0.5)

    path = plot_confusion(counts, 0.5, "essai", tmp_path / "cm.png")

    assert path.exists()
    assert path.stat().st_size > 1_000


# ===========================================================================
# Enregistrement complet
# ===========================================================================


def test_log_run_records_params_metrics_tags_and_artifacts(
    splits: dict, processed_dir: Path, client
) -> None:
    result = run_one("xgb", splits, seed=42)
    tracked = tracking.log_run(
        result, splits, client=client, processed_dir=processed_dir
    )

    run = client.get_run(tracked.run_id)

    # Parametres : les entrees
    assert run.data.params["model_kind"] == "xgb"
    assert run.data.params["seed"] == "42"
    assert "scale_pos_weight" in run.data.params

    # Metriques : prefixees, donc comparables fold par fold
    assert "val_pr_auc" in run.data.metrics
    assert "test_pr_auc" in run.data.metrics
    assert run.data.metrics["test_pr_auc"] == pytest.approx(
        result.test["pr_auc"], abs=1e-6
    )
    assert not any("accuracy" in key.lower() for key in run.data.metrics)

    # Tags : la tracabilite
    assert run.data.tags["dataset_sha256"] == "b" * 64
    assert run.data.tags["git_dirty"] in {"true", "false", "unknown"}
    assert run.data.tags["threshold_source"] == "validation"

    # Artefacts
    artifacts = {item.path for item in client.list_artifacts(tracked.run_id, "plots")}
    assert "plots/pr_curve.png" in artifacts
    assert "plots/confusion_matrix.png" in artifacts


def test_logged_model_is_registered_with_a_version(
    splits: dict, processed_dir: Path, client
) -> None:
    result = run_one("logreg", splits, seed=42)
    tracked = tracking.log_run(
        result, splits, client=client, processed_dir=processed_dir
    )

    assert tracked.version is not None
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    assert any(version.run_id == tracked.run_id for version in versions)


def test_model_reloads_through_the_alias_and_returns_scores(
    splits: dict, processed_dir: Path, client
) -> None:
    """Le contrat de la Phase 2, verifie de bout en bout.

    Le modele doit se recharger via l'ALIAS (pas un run_id) et renvoyer des
    SCORES, pas des etiquettes 0/1 : c'est le service qui applique le seuil.
    """
    result = run_one("xgb", splits, seed=42)
    tracked = tracking.log_run(
        result, splits, client=client, processed_dir=processed_dir
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL, PRODUCTION_ALIAS, tracked.version
    )

    loaded = mlflow.pyfunc.load_model(
        f"models:/{REGISTERED_MODEL}@{PRODUCTION_ALIAS}"
    )
    predictions = np.asarray(loaded.predict(splits["test"].X.head(5)))

    assert predictions.shape == (5, 2)  # [P(normal), P(fraude)]
    assert ((predictions >= 0.0) & (predictions <= 1.0)).all()
    expected = result.model.predict_proba(splits["test"].X.head(5))
    assert np.allclose(predictions, expected, atol=1e-6)


def test_signature_rejects_a_missing_column(
    splits: dict, processed_dir: Path, client
) -> None:
    """La signature est le filet de securite de l'API : une colonne absente
    doit faire ECHOUER l'appel, pas produire une prediction fausse en silence."""
    result = run_one("logreg", splits, seed=42)
    tracked = tracking.log_run(
        result, splits, client=client, processed_dir=processed_dir
    )
    loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{tracked.version}")

    truncated = splits["test"].X.head(3).drop(columns=["hour_of_day"])
    with pytest.raises(Exception, match="schema"):
        loaded.predict(truncated)


# ===========================================================================
# Promotion
# ===========================================================================


def _fake_result(kind: str, pr_auc: float) -> RunResult:
    """RunResult fabrique : la logique du gate ne doit dependre que du score.

    Faire tourner un vrai modele pour tester le plancher rendrait le test
    tributaire de ce que ce modele obtient sur des donnees synthetiques -- et
    c'est exactement ce qui l'a fait echouer la premiere fois, LogReg atteignant
    une PR-AUC quasi parfaite sur un signal lineaire.
    """
    return RunResult(
        model_kind=kind,
        seed=42,
        params={},
        val={"pr_auc": pr_auc},
        test={"pr_auc": pr_auc, "pr_auc_baseline": 0.002},
        frozen_threshold=0.5,
        threshold_attainable=True,
        fit_seconds=0.1,
        n_train_rows=600,
        n_train_positives=40,
    )


def test_promotion_refuses_a_model_below_the_floor() -> None:
    """Le plancher est la maquette du gate CI de la Phase 4 : un modele degrade
    ne doit pas pouvoir devenir la production.

    client=None est deliberé : le refus doit intervenir AVANT toute ecriture
    dans le registre. Si le code touchait le client, ce test planterait.
    """
    weak = _fake_result("logreg", pr_auc=0.42)
    tracked = {"logreg": TrackedRun("logreg", "run-id", "1")}

    with pytest.raises(TrackingError, match="Promotion refusee"):
        promote_best([weak], tracked, client=None, min_pr_auc=0.80)


def test_promotion_ranks_candidates_by_test_pr_auc() -> None:
    """Le meilleur est choisi sur la PR-AUC de TEST, pas sur l'ordre d'arrivee."""
    results = [
        _fake_result("logreg", 0.76),
        _fake_result("xgb", 0.87),
        _fake_result("xgb_smote", 0.86),
    ]
    tracked = {
        kind: TrackedRun(kind, f"run-{kind}", str(index + 1))
        for index, kind in enumerate(("logreg", "xgb", "xgb_smote"))
    }

    calls: list[tuple[str, str, str]] = []

    class FakeClient:
        def set_registered_model_alias(self, name, alias, version):
            calls.append((name, alias, version))

    promoted = promote_best(
        results, tracked, client=FakeClient(), min_pr_auc=0.80
    )

    assert promoted.model_kind == "xgb"
    assert calls == [(REGISTERED_MODEL, PRODUCTION_ALIAS, "2")]


def test_promotion_picks_the_best_run_by_test_pr_auc(
    splits: dict, processed_dir: Path, client
) -> None:
    results = []
    tracked = {}
    for kind in ("logreg", "xgb"):
        result = run_one(kind, splits, seed=42)
        results.append(result)
        tracked[kind] = tracking.log_run(
            result, splits, client=client, processed_dir=processed_dir
        )

    best_kind = max(results, key=lambda item: item.test["pr_auc"]).model_kind
    promoted = promote_best(results, tracked, client=client, min_pr_auc=0.0)

    assert promoted.model_kind == best_kind
    alias_target = client.get_model_version_by_alias(REGISTERED_MODEL, PRODUCTION_ALIAS)
    assert str(alias_target.version) == promoted.version


def test_promotion_is_never_automatic(splits: dict, processed_dir: Path, client) -> None:
    """log_run enregistre mais ne promeut pas : l'alias reste ou il etait."""
    result = run_one("xgb", splits, seed=42)
    tracking.log_run(result, splits, client=client, processed_dir=processed_dir)

    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(REGISTERED_MODEL, PRODUCTION_ALIAS)
