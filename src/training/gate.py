"""Lecture et application du gate de qualite (config/gate.yaml).

    python -m src.training.gate --report reports/training/xgb.json

Code de retour 0 si le modele passe le plancher, 1 sinon. C'est ce qui permet a
la Phase 4 de faire de ce controle une simple etape de CI, sans analyse de logs.

Le gate repond a UNE question : "ce modele est-il encore bon a expedier ?".
Il ne repond pas a "est-ce le meilleur qu'on ait eu ?" -- ca demanderait une
comparaison au modele @production courant plutot qu'un plancher fixe.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT

GATE_PATH: Path = PROJECT_ROOT / "config" / "gate.yaml"


class GateError(RuntimeError):
    """Gate illisible, incoherent, ou rapport introuvable."""


@dataclass(frozen=True)
class Gate:
    metric: str
    min_pr_auc: float
    baseline: dict[str, Any]

    @property
    def baseline_score(self) -> float | None:
        value = self.baseline.get(self.metric)
        return float(value) if value is not None else None


@dataclass(frozen=True)
class GateResult:
    passed: bool
    metric: str
    observed: float
    floor: float

    @property
    def margin(self) -> float:
        return self.observed - self.floor

    def render(self) -> str:
        bar = "=" * 66
        verdict = "PASSE" if self.passed else "BLOQUE"
        lines = [
            "",
            bar,
            f"  GATE QUALITE : {verdict}",
            bar,
            f"  metrique  : {self.metric}",
            f"  observe   : {self.observed:.4f}",
            f"  plancher  : {self.floor:.4f}",
            f"  marge     : {self.margin:+.4f}",
        ]
        if not self.passed:
            lines += [
                "",
                "  Le modele est sous le plancher de config/gate.yaml.",
                "  Soit la modification a degrade le modele, soit le plancher",
                "  doit bouger -- et le deplacer est un acte delibere, visible",
                "  en revue, pas un reglage discret.",
            ]
        lines += [bar, ""]
        return "\n".join(lines)


def load_gate(path: Path = GATE_PATH) -> Gate:
    """Charge et VALIDE config/gate.yaml."""
    if not path.exists():
        raise GateError(f"{path} est introuvable.")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GateError(f"{path.name} n'est pas un YAML valide : {exc}") from exc

    if not isinstance(raw, dict) or "gate" not in raw:
        raise GateError(f"{path.name} ne contient pas de section 'gate'.")

    section = raw["gate"]
    for key in ("metric", "min_pr_auc"):
        if key not in section:
            raise GateError(f"{path.name} : la cle 'gate.{key}' est absente.")

    floor = float(section["min_pr_auc"])
    if not 0.0 < floor < 1.0:
        raise GateError(
            f"min_pr_auc doit etre dans ]0, 1[, valeur lue : {floor}. "
            "Une PR-AUC est une aire, pas un pourcentage."
        )

    gate = Gate(
        metric=str(section["metric"]),
        min_pr_auc=floor,
        baseline=dict(raw.get("baseline") or {}),
    )

    # Invariant : un plancher au-dessus de la baseline enregistree rendrait la
    # CI rouge des le premier jour, sur le modele meme qui a servi a l'etablir.
    recorded = gate.baseline_score
    if recorded is not None and floor >= recorded:
        raise GateError(
            f"Gate incoherent : le plancher ({floor:.4f}) est superieur ou egal "
            f"a la baseline enregistree ({recorded:.4f}). La CI bloquerait le "
            "modele qui a servi a fixer le plancher."
        )
    return gate


def read_metric(report_path: Path, metric: str) -> float:
    """Extrait la metrique d'un rapport ecrit par src.training.train.

    ``test_pr_auc`` designe la cle ``pr_auc`` du bloc ``test`` : c'est le
    prefixage utilise cote MLflow, on le reutilise ici pour que le nom du gate
    soit le meme partout.
    """
    if not report_path.exists():
        raise GateError(
            f"{report_path} est introuvable.\n"
            "  Lance d'abord un entrainement :  python -m src.training.train"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    split, _, key = metric.partition("_")
    if split not in report or not key:
        raise GateError(
            f"Metrique {metric!r} illisible dans {report_path.name}. "
            "Format attendu : <split>_<metrique>, par ex. test_pr_auc."
        )
    if key not in report[split]:
        raise GateError(
            f"{report_path.name} ne contient pas {key!r} dans le bloc {split!r}."
        )
    return float(report[split][key])


def check(report_path: Path, gate: Gate | None = None) -> GateResult:
    gate = gate or load_gate()
    observed = read_metric(report_path, gate.metric)
    return GateResult(
        passed=observed >= gate.min_pr_auc,
        metric=gate.metric,
        observed=observed,
        floor=gate.min_pr_auc,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.gate",
        description="Applique le plancher de config/gate.yaml a un rapport d'entrainement.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="rapport JSON produit par src.training.train",
    )
    parser.add_argument("--gate", type=Path, default=GATE_PATH)
    args = parser.parse_args(argv)

    try:
        result = check(args.report, load_gate(args.gate))
    except GateError as exc:
        print(f"\nECHEC : {exc}\n", file=sys.stderr)
        return 1

    print(result.render())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
