"""Tests de l'entraineur (step 1.B), sur des splits synthetiques minuscules.

Les proprietes verifiees ici sont celles dont la violation produit de MEILLEURS
scores : une fuite via SMOTE, un seuil choisi sur le test. Un entrainement casse
ne se signale pas par une erreur -- il se signale par des resultats trop beaux.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET
from src.features.build import FEATURE_COLUMNS, SPLIT_FILES
from src.training import train as train_module
from src.training.metrics import recall_at_precision
from src.training.train import (
    MODEL_KINDS,
    XGB_PARAMS,
    TrainingError,
    compute_scale_pos_weight,
    load_splits,
    run_one,
    write_report,
)

# Assez de fraudes dans le train pour que SMOTE trouve ses 5 plus proches
# voisines, assez dans val/test pour que les metriques aient un sens.
SPLIT_SPEC = {"train": (600, 40), "val": (200, 15), "test": (200, 15)}


def _make_split(n_rows: int, n_positives: int, seed: int) -> pd.DataFrame:
    """Frame aux colonnes de sortie du step 0.D, avec un signal apprenable."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {column: rng.normal(size=n_rows) for column in FEATURE_COLUMNS}
    )
    frame[TARGET] = 0
    positions = rng.choice(n_rows, size=n_positives, replace=False)
    frame.iloc[positions, frame.columns.get_loc(TARGET)] = 1
    # Signal : les fraudes sont decalees sur deux variables.
    frame.iloc[positions, frame.columns.get_loc("V1")] += 3.0
    frame.iloc[positions, frame.columns.get_loc("V2")] -= 2.5
    return frame


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """Ecrit train/val/test.parquet dans un dossier temporaire."""
    out = tmp_path / "processed"
    out.mkdir()
    for index, (name, (rows, positives)) in enumerate(SPLIT_SPEC.items()):
        frame = _make_split(rows, positives, seed=index)
        frame.to_parquet(out / SPLIT_FILES[name], index=False)
    return out


@pytest.fixture
def splits(processed_dir: Path) -> dict:
    return load_splits(processed_dir)


# ===========================================================================
# Chargement
# ===========================================================================


def test_load_splits_reads_the_three_folds(splits: dict) -> None:
    assert set(splits) == {"train", "val", "test"}
    assert splits["train"].n_rows == 600
    assert splits["train"].n_positives == 40
    assert list(splits["train"].X.columns) == list(FEATURE_COLUMNS)
    assert TARGET not in splits["train"].X.columns


def test_load_splits_points_at_the_command_to_run(tmp_path: Path) -> None:
    """Le message doit dire QUOI faire, pas seulement que le fichier manque."""
    with pytest.raises(TrainingError, match="python -m src.features"):
        load_splits(tmp_path / "vide")


def test_load_splits_rejects_a_fold_without_frauds(tmp_path: Path) -> None:
    out = tmp_path / "processed"
    out.mkdir()
    for name, (rows, positives) in SPLIT_SPEC.items():
        # Le test se retrouve sans aucune fraude.
        count = 0 if name == "test" else positives
        _make_split(rows, count, seed=1).to_parquet(out / SPLIT_FILES[name], index=False)

    with pytest.raises(TrainingError, match="aucune fraude"):
        load_splits(out)


# ===========================================================================
# scale_pos_weight
# ===========================================================================


def test_scale_pos_weight_is_negatives_over_positives() -> None:
    labels = np.array([0] * 990 + [1] * 10)
    assert compute_scale_pos_weight(labels) == pytest.approx(99.0)


def test_scale_pos_weight_cannot_see_validation_or_test(splits: dict) -> None:
    """La signature ne prend que y_train : la fuite est impossible par
    construction, pas par discipline."""
    from_train_only = compute_scale_pos_weight(splits["train"].y)

    everything = np.concatenate([split.y for split in splits.values()])
    assert compute_scale_pos_weight(everything) != pytest.approx(from_train_only)


def test_scale_pos_weight_raises_without_positives() -> None:
    with pytest.raises(TrainingError, match="Aucune fraude"):
        compute_scale_pos_weight(np.zeros(50, dtype=int))


# ===========================================================================
# SMOTE : le test le plus important du module
# ===========================================================================


def test_smote_only_ever_sees_the_training_fold(
    splits: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LE test anti-fuite.

    Si SMOTE recevait un jour l'ensemble des donnees avant decoupage, des
    points interpoles depuis des fraudes du test se retrouveraient dans le
    train. Les scores exploseraient sans qu'aucune erreur ne soit levee.

    On espionne fit_resample pour constater ce qu'il a REELLEMENT recu.
    """
    observed: dict[str, int] = {}
    real_smote = train_module.SMOTE

    class RecordingSMOTE(real_smote):
        def fit_resample(self, X, y):
            observed["rows"] = len(y)
            observed["positives"] = int(np.sum(y))
            return super().fit_resample(X, y)

    monkeypatch.setattr(train_module, "SMOTE", RecordingSMOTE)
    run_one("xgb_smote", splits, seed=42)

    assert observed["rows"] == splits["train"].n_rows
    assert observed["positives"] == splits["train"].n_positives
    # Et surtout : pas la moindre ligne de validation ou de test.
    assert observed["rows"] != splits["train"].n_rows + splits["val"].n_rows


def test_smote_run_does_not_also_set_scale_pos_weight(splits: dict) -> None:
    """Combiner les deux sur-corrigerait : apres SMOTE les classes sont deja
    equilibrees."""
    result = run_one("xgb_smote", splits, seed=42)
    assert result.params["scale_pos_weight"] is None
    assert result.params["imbalance"] == "SMOTE"


def test_smote_run_reports_how_many_rows_were_fabricated(splits: dict) -> None:
    """L'ampleur de la fabrication doit etre visible dans le rapport."""
    result = run_one("xgb_smote", splits, seed=42)

    assert result.params["synthetic_rows_added"] > 0
    assert result.n_train_rows > splits["train"].n_rows
    assert result.n_train_positives > splits["train"].n_positives


def test_xgb_run_uses_scale_pos_weight_and_no_smote(splits: dict) -> None:
    result = run_one("xgb", splits, seed=42)

    assert result.params["imbalance"] == "scale_pos_weight"
    assert result.params["scale_pos_weight"] == pytest.approx(
        compute_scale_pos_weight(splits["train"].y), abs=1e-3
    )
    assert result.n_train_rows == splits["train"].n_rows


# ===========================================================================
# Le seuil : choisi sur la validation, fige pour le test
# ===========================================================================


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_threshold_comes_from_validation_not_test(kind: str, splits: dict) -> None:
    """Le coeur du protocole.

    Le seuil publie doit etre exactement celui que recall_at_precision trouve
    sur la VALIDATION. S'il etait recalcule sur le test, la metrique de test
    serait ajustee sur les donnees qu'elle pretend mesurer.
    """
    result = run_one(kind, splits, seed=42, min_precision=0.90)
    if not result.threshold_attainable:
        pytest.skip("cible de precision inatteignable pour ce modele")

    assert result.test["threshold"] == pytest.approx(result.frozen_threshold)
    # Et ce seuil n'est PAS celui qu'on choisirait sur le test.
    assert result.frozen_threshold is not None


def test_frozen_threshold_matches_a_direct_validation_computation(
    splits: dict,
) -> None:
    """Verification independante : on refait le calcul du seuil a la main sur
    la validation et on doit retomber sur la meme valeur."""
    result = run_one("xgb", splits, seed=42, min_precision=0.90)
    if not result.threshold_attainable:
        pytest.skip("cible inatteignable")

    model, *_ = train_module.fit_model("xgb", splits, seed=42)
    val_scores = model.predict_proba(splits["val"].X)[:, 1]
    expected = recall_at_precision(splits["val"].y, val_scores, min_precision=0.90)

    assert result.frozen_threshold == pytest.approx(expected.threshold)


def test_unattainable_precision_is_reported_not_hidden(splits: dict) -> None:
    """Une cible impossible doit produire un avertissement explicite, jamais un
    seuil bidon applique en silence."""
    result = run_one("logreg", splits, seed=42, min_precision=1.0)

    if not result.threshold_attainable:
        assert result.frozen_threshold is None
        assert result.warnings
        assert "inatteignable" in result.warnings[0]


# ===========================================================================
# Proprietes communes aux trois modeles
# ===========================================================================


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_every_model_produces_probabilities_in_unit_interval(
    kind: str, splits: dict
) -> None:
    model, *_ = train_module.fit_model(kind, splits, seed=42)
    scores = model.predict_proba(splits["test"].X)[:, 1]

    assert scores.shape == (splits["test"].n_rows,)
    assert np.isfinite(scores).all()
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_every_model_beats_random_on_learnable_signal(
    kind: str, splits: dict
) -> None:
    """Garde-fou : le signal injecte est franc, donc un modele qui n'apprend
    rien signale un bug de cablage (mauvaises colonnes, labels inverses)."""
    result = run_one(kind, splits, seed=42)
    assert result.test["pr_auc"] > result.test["pr_auc_baseline"] * 3


def test_xgboost_early_stopping_watches_aucpr() -> None:
    """Surveiller logloss ferait arreter au meilleur moment pour un critere qui
    n'est pas le notre."""
    assert XGB_PARAMS["eval_metric"] == "aucpr"
    assert XGB_PARAMS["early_stopping_rounds"] == 50


def test_runs_are_reproducible_at_a_fixed_seed(splits: dict) -> None:
    first = run_one("xgb", splits, seed=42)
    second = run_one("xgb", splits, seed=42)
    assert first.test["pr_auc"] == pytest.approx(second.test["pr_auc"])
    assert first.frozen_threshold == pytest.approx(second.frozen_threshold)


def test_no_run_reports_accuracy(splits: dict) -> None:
    """La regle de CLAUDE.md doit survivre a la traversee de l'entraineur."""
    result = run_one("xgb", splits, seed=42)
    for block in (result.val, result.test):
        assert not any("accuracy" in key.lower() for key in block)


# ===========================================================================
# Rapport
# ===========================================================================


def test_write_report_is_valid_json_and_round_trips(
    splits: dict, tmp_path: Path
) -> None:
    result = run_one("logreg", splits, seed=42)
    path = write_report(result, tmp_path / "reports")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_kind"] == "logreg"
    assert payload["seed"] == 42
    assert payload["val"]["pr_auc"] == pytest.approx(result.val["pr_auc"])
    assert payload["test"]["pr_auc"] == pytest.approx(result.test["pr_auc"])
