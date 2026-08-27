"""Entrainement et comparaison des trois strategies de desequilibre.

    python -m src.training.train                # les trois runs
    python -m src.training.train --model xgb    # un seul

Un modele ne minimise jamais la metrique qu'on lui reclame : il minimise une
fonction de cout DERIVABLE, la log-loss. Or la log-loss brute additionne
l'erreur sur les 284 807 transactions, dont 0,17 % seulement sont des fraudes :
le moyen le plus rentable de la faire baisser est d'ignorer les fraudes. Le
modele n'a pas tort, il obeit au contrat qu'on lui a donne.

Les trois runs sont trois facons de reecrire ce contrat :

  logreg      class_weight="balanced"  -> repondere la log-loss (~289x)
  xgb         scale_pos_weight ~= 577  -> repondere gradients et hessiennes
  xgb_smote   SMOTE sur le TRAIN seul  -> fabriquer des fraudes synthetiques

Protocole commun : entrainement sur TRAIN, choix du seuil sur VALIDATION, puis
evaluation du TEST **a ce seuil fige**. Ajuster le seuil sur le test avant de
publier une metrique sur ce meme test revient a se noter sur sa propre copie.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import DATA_PROCESSED, PROJECT_ROOT, TARGET
from src.features.build import FEATURE_COLUMNS, SPLIT_FILES
from src.training.metrics import (
    DEFAULT_MIN_PRECISION,
    evaluate,
    recall_at_precision,
)

ModelKind = Literal["logreg", "xgb", "xgb_smote"]
MODEL_KINDS: tuple[str, ...] = ("logreg", "xgb", "xgb_smote")

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "training"

# Parametres XGBoost communs aux deux variantes.
#   eval_metric="aucpr" : c'est ICI qu'on rebranche le substitut derivable sur
#   la metrique qui nous interesse. Surveiller "logloss" ferait arreter
#   l'entrainement au meilleur moment pour un critere qui n'est pas le notre.
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50,
}


class TrainingError(RuntimeError):
    """Entrainement impossible (donnees absentes ou incoherentes)."""


@dataclass(frozen=True)
class Split:
    name: str
    X: pd.DataFrame
    y: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.y.size)

    @property
    def n_positives(self) -> int:
        return int((self.y == 1).sum())


@dataclass
class RunResult:
    model_kind: str
    seed: int
    params: dict[str, Any]
    val: dict[str, float]
    test: dict[str, float]
    frozen_threshold: float | None
    threshold_attainable: bool
    fit_seconds: float
    n_train_rows: int
    n_train_positives: int
    best_iteration: int | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


def load_splits(processed_dir: Path = DATA_PROCESSED) -> dict[str, Split]:
    """Charge les trois parquets ecrits par ``python -m src.features``."""
    splits: dict[str, Split] = {}
    expected = [*FEATURE_COLUMNS, TARGET]

    for name, filename in SPLIT_FILES.items():
        path = processed_dir / filename
        if not path.exists():
            raise TrainingError(
                f"{path} est introuvable.\n"
                "  Lance d'abord le decoupage :  python -m src.features"
            )
        frame = pd.read_parquet(path)
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise TrainingError(
                f"{path.name} ne contient pas les colonnes attendues.\n"
                f"  Manquantes : {missing}\n"
                "  Le decoupage a probablement ete produit par une version "
                "anterieure du pipeline : relance python -m src.features"
            )
        splits[name] = Split(
            name=name,
            X=frame[list(FEATURE_COLUMNS)],
            y=frame[TARGET].to_numpy(dtype=int),
        )

    for name, split in splits.items():
        if split.n_positives == 0:
            raise TrainingError(
                f"Le split {name} ne contient aucune fraude : impossible "
                "d'evaluer. Verifie la stratification du decoupage."
            )
    return splits


# ---------------------------------------------------------------------------
# Construction des modeles
# ---------------------------------------------------------------------------


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """n_negatifs / n_positifs, calcule sur le TRAIN uniquement.

    La signature ne prend volontairement que y_train : la fonction est dans
    l'incapacite de voir la validation ou le test, meme par accident.

    Effet : la masse totale des fraudes egale celle des transactions normales
    dans le gradient. Aucune donnee dupliquee, aucun point invente.

    Contrepartie : les probabilites sorties ne sont PLUS CALIBREES. Le modele
    repond "probabilite si la fraude representait 50 % du trafic". On ne se sert
    que du classement (PR-AUC) et d'un seuil empirique, donc ca ne gene pas ici
    -- mais l'API de la Phase 2 ne devra pas presenter ces scores comme des
    probabilites reelles.
    """
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        raise TrainingError("Aucune fraude dans le train : scale_pos_weight indefini.")
    return negatives / positives


def build_logreg(seed: int) -> Pipeline:
    """Regression logistique, scaler INCLUS dans le pipeline.

    Le StandardScaler vit ici et non dans le pipeline de variables : ainsi il
    est ajuste sur le train seul par construction, et le serving charge un seul
    objet qui normalise ET predit. Oublier de normaliser a l'inference devient
    impossible.

    XGBoost n'en a pas besoin : une coupure d'arbre "x > 3.7" ne change pas si
    l'on multiplie x par 1000. La descente de gradient de la regression, elle,
    rame sur une surface d'erreur etiree.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_xgb(seed: int, scale_pos_weight: float | None) -> XGBClassifier:
    return XGBClassifier(
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        **XGB_PARAMS,
    )


# ---------------------------------------------------------------------------
# Entrainement
# ---------------------------------------------------------------------------


def fit_model(
    kind: str,
    splits: dict[str, Split],
    *,
    seed: int,
    smote_ratio: float | str = "auto",
) -> tuple[Any, dict[str, Any], int, int, int | None]:
    """Entraine un modele et renvoie (modele, params, n_lignes, n_fraudes, best_iter).

    ``n_lignes`` et ``n_fraudes`` decrivent les donnees REELLEMENT vues par le
    modele : pour xgb_smote elles refletent l'echantillon reequilibre, ce qui
    rend l'ampleur de la fabrication visible dans le rapport.
    """
    train, val = splits["train"], splits["val"]

    if kind == "logreg":
        model = build_logreg(seed)
        model.fit(train.X, train.y)
        params = {
            "model": "logistic_regression",
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 2000,
            "scaler": "StandardScaler (dans le pipeline)",
        }
        return model, params, train.n_rows, train.n_positives, None

    if kind == "xgb":
        weight = compute_scale_pos_weight(train.y)
        model = build_xgb(seed, weight)
        # eval_set = la VRAIE validation : c'est elle qui pilote l'arret
        # anticipe, et elle n'est jamais reechantillonnee.
        model.fit(train.X, train.y, eval_set=[(val.X, val.y)], verbose=False)
        params = {
            "model": "xgboost",
            "imbalance": "scale_pos_weight",
            "scale_pos_weight": round(weight, 3),
            **XGB_PARAMS,
        }
        return model, params, train.n_rows, train.n_positives, _best_iteration(model)

    if kind == "xgb_smote":
        # SMOTE UNIQUEMENT sur le train, APRES le decoupage. Applique avant, un
        # point interpole depuis une fraude du test se retrouverait dans le
        # train : des quasi-doublons des deux cotes, des scores spectaculaires,
        # et aucune signification. C'est l'erreur la plus repandue sur ce
        # dataset.
        sampler = SMOTE(random_state=seed, sampling_strategy=smote_ratio)
        X_resampled, y_resampled = sampler.fit_resample(train.X, train.y)

        # PAS de scale_pos_weight ici : les classes sont deja reequilibrees,
        # y ajouter un poids de 577 sur-corrigerait dans l'autre sens.
        model = build_xgb(seed, None)
        model.fit(
            X_resampled, y_resampled, eval_set=[(val.X, val.y)], verbose=False
        )
        params = {
            "model": "xgboost",
            "imbalance": "SMOTE",
            "smote_sampling_strategy": smote_ratio,
            "smote_k_neighbors": 5,
            "scale_pos_weight": None,
            "synthetic_rows_added": int(len(y_resampled) - train.n_rows),
            **XGB_PARAMS,
        }
        return (
            model,
            params,
            len(y_resampled),
            int((y_resampled == 1).sum()),
            _best_iteration(model),
        )

    raise TrainingError(f"Modele inconnu : {kind!r}. Choix : {MODEL_KINDS}")


def _best_iteration(model: XGBClassifier) -> int | None:
    value = getattr(model, "best_iteration", None)
    return int(value) if value is not None else None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_one(
    kind: str,
    splits: dict[str, Split],
    *,
    seed: int = 42,
    min_precision: float = DEFAULT_MIN_PRECISION,
    smote_ratio: float | str = "auto",
) -> RunResult:
    """Entraine, choisit le seuil sur la validation, evalue le test a ce seuil."""
    started = time.perf_counter()
    model, params, n_rows, n_positives, best_iteration = fit_model(
        kind, splits, seed=seed, smote_ratio=smote_ratio
    )
    fit_seconds = time.perf_counter() - started

    val, test = splits["val"], splits["test"]
    val_scores = model.predict_proba(val.X)[:, 1]
    test_scores = model.predict_proba(test.X)[:, 1]

    # 1) le seuil se decide sur la VALIDATION
    found = recall_at_precision(val.y, val_scores, min_precision=min_precision)
    val_metrics = evaluate(val.y, val_scores, min_precision=min_precision)

    warnings: list[str] = []
    if found.attainable:
        # 2) puis il est FIGE et impose au test
        test_metrics = evaluate(
            test.y, test_scores, threshold=found.threshold, min_precision=min_precision
        )
    else:
        warnings.append(
            f"Precision de {min_precision:.0%} inatteignable sur la validation "
            f"(meilleure : {found.achieved_precision:.1%}). Le test est evalue "
            "sans seuil impose."
        )
        test_metrics = evaluate(test.y, test_scores, min_precision=min_precision)

    return RunResult(
        model_kind=kind,
        seed=seed,
        params=params,
        val=val_metrics,
        test=test_metrics,
        frozen_threshold=found.threshold,
        threshold_attainable=found.attainable,
        fit_seconds=fit_seconds,
        n_train_rows=n_rows,
        n_train_positives=n_positives,
        best_iteration=best_iteration,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Restitution
# ---------------------------------------------------------------------------


def render_run(result: RunResult, min_precision: float) -> str:
    target = f"recall_at_p{round(min_precision * 100)}"
    bar = "-" * 74
    lines = [
        bar,
        f"  {result.model_kind}",
        bar,
        f"    donnees vues        : {result.n_train_rows:,} lignes, "
        f"{result.n_train_positives:,} fraudes".replace(",", " "),
        f"    duree entrainement  : {result.fit_seconds:.1f} s",
    ]
    if result.best_iteration is not None:
        lines.append(
            f"    arbres retenus      : {result.best_iteration + 1} "
            f"/ {XGB_PARAMS['n_estimators']} (arret anticipe)"
        )
    lines += [
        "",
        (
            f"    VALIDATION  PR-AUC {result.val['pr_auc']:.4f}   "
            f"ROC-AUC {result.val['roc_auc']:.4f}   "
            f"{target} {result.val[target]:.4f}"
        ),
    ]
    if result.threshold_attainable:
        lines.append(f"    seuil fige sur la validation : {result.frozen_threshold:.6f}")
    lines.append(
        f"    TEST        PR-AUC {result.test['pr_auc']:.4f}   "
        f"ROC-AUC {result.test['roc_auc']:.4f}   "
        f"{target} {result.test[target]:.4f}"
    )
    if "tp" in result.test:
        lines.append(
            f"    au seuil fige : TP {int(result.test['tp'])}  "
            f"FP {int(result.test['fp'])}  FN {int(result.test['fn'])}  "
            f"-> precision {result.test['precision_at_threshold']:.1%}, "
            f"rappel {result.test['recall_at_threshold']:.1%}"
        )
    for warning in result.warnings:
        lines.append(f"    ATTENTION : {warning}")
    lines.append("")
    return "\n".join(lines)


def render_comparison(results: list[RunResult], min_precision: float) -> str:
    target = f"recall_at_p{round(min_precision * 100)}"
    bar = "=" * 74
    lines = [
        "",
        bar,
        "  COMPARAISON  (seuil choisi sur VALIDATION, fige pour TEST)",
        bar,
        (
            f"  {'modele':<12}{'PR-AUC':>9}{'x hasard':>10}{'ROC-AUC':>9}"
            f"{target.replace('recall_at_', 'rec@'):>9}{'TP':>5}{'FP':>5}{'FN':>5}"
        ),
        "  " + "-" * 70,
    ]
    for result in sorted(results, key=lambda item: item.test["pr_auc"], reverse=True):
        test = result.test
        counts = (
            f"{int(test['tp']):>5}{int(test['fp']):>5}{int(test['fn']):>5}"
            if "tp" in test
            else f"{'-':>5}{'-':>5}{'-':>5}"
        )
        lines.append(
            f"  {result.model_kind:<12}{test['pr_auc']:>9.4f}"
            f"{test['pr_auc_lift']:>9.0f}x{test['roc_auc']:>9.4f}"
            f"{test[target]:>9.4f}{counts}"
        )
    baseline = results[0].test["pr_auc_baseline"]
    lines += [
        "  " + "-" * 70,
        f"  PR-AUC du hasard : {baseline:.6f}  (une PR-AUC ne se lit que",
        "  comparee a cette reference, pas sur l'echelle du ROC)",
        bar,
        "",
    ]
    return "\n".join(lines)


def write_report(result: RunResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.model_kind}.json"
    payload = {
        "model_kind": result.model_kind,
        "seed": result.seed,
        "params": result.params,
        "frozen_threshold": result.frozen_threshold,
        "threshold_attainable": result.threshold_attainable,
        "fit_seconds": round(result.fit_seconds, 3),
        "n_train_rows": result.n_train_rows,
        "n_train_positives": result.n_train_positives,
        "best_iteration": result.best_iteration,
        "warnings": result.warnings,
        "val": result.val,
        "test": result.test,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _smote_ratio(raw: str) -> float | str:
    if raw == "auto":
        return "auto"
    try:
        return float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--smote-ratio attend 'auto' ou un nombre, recu {raw!r}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.train",
        description="Entraine et compare les trois strategies de desequilibre.",
    )
    parser.add_argument(
        "--model",
        choices=(*MODEL_KINDS, "all"),
        default="all",
        help="modele a entrainer (defaut : all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-precision",
        type=float,
        default=DEFAULT_MIN_PRECISION,
        help="precision cible pour le choix du seuil (defaut : 0.90)",
    )
    parser.add_argument(
        "--smote-ratio",
        type=_smote_ratio,
        default="auto",
        help=(
            "'auto' equilibre completement les classes (~170 000 points "
            "synthetiques a partir de 295 fraudes). Un nombre, par ex. 0.1, "
            "limite la fabrication."
        ),
    )
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args(argv)

    try:
        splits = load_splits(args.processed_dir)
    except TrainingError as exc:
        print(f"\nECHEC : {exc}\n", file=sys.stderr)
        return 1

    print(
        "\nSplits charges : "
        + ", ".join(
            f"{name} {split.n_rows:,} lignes / {split.n_positives} fraudes".replace(
                ",", " "
            )
            for name, split in splits.items()
        )
    )
    print(f"Seuil choisi sur la validation a >= {args.min_precision:.0%} de precision.\n")

    kinds = MODEL_KINDS if args.model == "all" else (args.model,)
    results: list[RunResult] = []
    for kind in kinds:
        print(f"[train] {kind} ...")
        try:
            result = run_one(
                kind,
                splits,
                seed=args.seed,
                min_precision=args.min_precision,
                smote_ratio=args.smote_ratio,
            )
        except TrainingError as exc:
            print(f"\nECHEC sur {kind} : {exc}\n", file=sys.stderr)
            return 1
        results.append(result)
        print(render_run(result, args.min_precision))
        write_report(result, args.out_dir)

    if len(results) > 1:
        print(render_comparison(results, args.min_precision))
    print(f"Rapports JSON ecrits dans {args.out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
