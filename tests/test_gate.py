"""Tests du gate de qualite (step 1.E).

Le gate est le seul mecanisme qui empechera un modele degrade d'etre expedie en
Phase 4. Un gate casse ne se signale pas : il laisse simplement tout passer.
D'ou les tests sur les DEUX verdicts, et sur les incoherences de configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.training.gate import (
    GATE_PATH,
    Gate,
    GateError,
    check,
    load_gate,
    read_metric,
)


def _write_gate(path: Path, *, floor: float = 0.80, baseline: float | None = 0.8727):
    payload: dict = {
        "version": 1,
        "gate": {"metric": "test_pr_auc", "min_pr_auc": floor},
    }
    if baseline is not None:
        payload["baseline"] = {"model": "xgb", "test_pr_auc": baseline}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_report(path: Path, pr_auc: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "model_kind": "xgb",
                "val": {"pr_auc": pr_auc - 0.05},
                "test": {"pr_auc": pr_auc, "roc_auc": 0.98},
            }
        ),
        encoding="utf-8",
    )
    return path


# ===========================================================================
# Le fichier reel du projet
# ===========================================================================


def test_project_gate_file_is_valid() -> None:
    """config/gate.yaml doit rester lisible : il est la reference partagee
    entre un entrainement local et la CI."""
    gate = load_gate()

    assert gate.metric == "test_pr_auc"
    assert gate.min_pr_auc == pytest.approx(0.80)


def test_project_gate_floor_sits_below_the_recorded_baseline() -> None:
    """L'invariant central.

    Un plancher au-dessus de la baseline rendrait la CI rouge des le premier
    jour, sur le modele meme qui a servi a l'etablir. C'est le genre d'erreur
    qui ne se voit qu'au moment ou la CI casse, en Phase 4.
    """
    gate = load_gate()

    assert gate.baseline_score is not None
    assert gate.min_pr_auc < gate.baseline_score


def test_project_gate_rejects_the_linear_baseline() -> None:
    """Le plancher doit bloquer un retour a logreg (0.7602 mesure), sinon un
    modele ayant perdu tout l'apport de XGBoost passerait."""
    assert load_gate().min_pr_auc > 0.7602


def test_project_gate_passes_the_worst_observed_good_fold() -> None:
    """Et il doit rester sous le PIRE fold du BON modele (val 0.8234), sinon la
    CI rejetterait le modele qu'on veut garder."""
    assert load_gate().min_pr_auc < 0.8234


def test_gate_path_is_versioned_in_the_repo() -> None:
    assert GATE_PATH.exists()
    assert GATE_PATH.parts[-2:] == ("config", "gate.yaml")


# ===========================================================================
# Chargement et validation
# ===========================================================================


def test_missing_gate_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="introuvable"):
        load_gate(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text("gate: [unclosed", encoding="utf-8")

    with pytest.raises(GateError, match="YAML valide"):
        load_gate(path)


def test_missing_gate_section_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text(yaml.safe_dump({"version": 1}), encoding="utf-8")

    with pytest.raises(GateError, match="section 'gate'"):
        load_gate(path)


def test_missing_key_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text(yaml.safe_dump({"gate": {"metric": "test_pr_auc"}}), encoding="utf-8")

    with pytest.raises(GateError, match="min_pr_auc"):
        load_gate(path)


@pytest.mark.parametrize("floor", [0.0, 1.0, 80.0, -0.5])
def test_floor_outside_zero_one_is_rejected(tmp_path: Path, floor: float) -> None:
    """80 au lieu de 0.80 est l'erreur de saisie evidente : une PR-AUC est une
    aire, pas un pourcentage."""
    path = _write_gate(tmp_path / "gate.yaml", floor=floor, baseline=None)

    with pytest.raises(GateError, match="min_pr_auc"):
        load_gate(path)


def test_floor_above_baseline_is_rejected(tmp_path: Path) -> None:
    """Incoherence : la CI bloquerait le modele qui a fixe le plancher."""
    path = _write_gate(tmp_path / "gate.yaml", floor=0.95, baseline=0.8727)

    with pytest.raises(GateError, match="incoherent"):
        load_gate(path)


def test_gate_without_baseline_is_accepted(tmp_path: Path) -> None:
    """La baseline est de la provenance, pas une obligation : son absence ne
    doit pas empecher la CI de tourner."""
    path = _write_gate(tmp_path / "gate.yaml", floor=0.80, baseline=None)

    assert load_gate(path).baseline_score is None


# ===========================================================================
# Lecture du rapport
# ===========================================================================


def test_read_metric_resolves_split_and_key(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "xgb.json", pr_auc=0.87)

    assert read_metric(report, "test_pr_auc") == pytest.approx(0.87)
    assert read_metric(report, "val_pr_auc") == pytest.approx(0.82)


def test_missing_report_points_at_the_command_to_run(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="python -m src.training.train"):
        read_metric(tmp_path / "absent.json", "test_pr_auc")


def test_unknown_metric_in_report_is_reported(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "xgb.json", pr_auc=0.87)

    with pytest.raises(GateError, match="ne contient pas"):
        read_metric(report, "test_f1")


# ===========================================================================
# Les deux verdicts
# ===========================================================================


def test_model_above_the_floor_passes(tmp_path: Path) -> None:
    gate = Gate(metric="test_pr_auc", min_pr_auc=0.80, baseline={})
    report = _write_report(tmp_path / "xgb.json", pr_auc=0.8727)

    result = check(report, gate)

    assert result.passed is True
    assert result.margin == pytest.approx(0.0727)
    assert "PASSE" in result.render()


def test_model_below_the_floor_is_blocked(tmp_path: Path) -> None:
    """Le verdict qui compte : sans lui, le gate laisse tout passer et personne
    ne s'en apercoit."""
    gate = Gate(metric="test_pr_auc", min_pr_auc=0.80, baseline={})
    report = _write_report(tmp_path / "degrade.json", pr_auc=0.6100)

    result = check(report, gate)

    assert result.passed is False
    assert result.margin < 0
    assert "BLOQUE" in result.render()


def test_model_exactly_at_the_floor_passes(tmp_path: Path) -> None:
    """Comparaison >= : le plancher est inclusif, sinon la valeur du plancher
    elle-meme serait rejetee."""
    gate = Gate(metric="test_pr_auc", min_pr_auc=0.80, baseline={})
    report = _write_report(tmp_path / "pile.json", pr_auc=0.80)

    assert check(report, gate).passed is True


def test_the_real_baseline_would_pass_the_real_gate(tmp_path: Path) -> None:
    """Bout en bout sur les vraies valeurs : le modele promu en @production
    passe le gate qu'il a lui-meme servi a fixer."""
    gate = load_gate()
    report = _write_report(tmp_path / "xgb.json", pr_auc=gate.baseline_score)

    assert check(report, gate).passed is True


def test_the_linear_baseline_would_be_blocked(tmp_path: Path) -> None:
    """Contre-epreuve : logreg (0.7602 mesure) doit etre refuse."""
    gate = load_gate()
    report = _write_report(tmp_path / "logreg.json", pr_auc=0.7602)

    assert check(report, gate).passed is False
