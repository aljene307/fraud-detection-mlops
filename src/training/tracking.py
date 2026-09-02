"""Enregistrement des runs dans MLflow : tracking, artefacts, registre, alias.

Ce module est SEPARE de train.py a dessein : ``run_one()`` reste executable sans
serveur de tracking, et les tests d'entrainement n'ont pas besoin d'une base
MLflow. La dependance ne va que dans un sens -- train.py importe tracking.py,
jamais l'inverse.

Vocabulaire MLflow :

* **experience** : un dossier nomme qui regroupe des runs comparables. La
  comparaison dans l'interface n'a de sens qu'a l'interieur d'une experience.
* **run** : une execution, identifiee par un run_id. Elle porte des parametres
  (les entrees, immuables, sur lesquelles on FILTRE), des metriques (les sorties
  numeriques, sur lesquelles on TRIE), des tags (metadonnees modifiables) et des
  artefacts (des fichiers).
* **signature** : le schema declare des entrees et sorties du modele. Elle est
  APPLIQUEE au rechargement : une colonne manquante fait echouer l'appel au lieu
  de produire une prediction silencieusement fausse.
* **registre** : un catalogue nomme au-dessus des runs, avec des versions.
* **alias** : un pointeur mutable vers une version. Le serving charge
  ``models:/fraud-detector@production`` ; revenir en arriere consiste a
  repointer l'alias, sans redeploiement ni changement de code.

Pourquoi PAS ``mlflow.autolog()`` : il journalise automatiquement
``training_accuracy_score``. Il violerait la regle de CLAUDE.md des la premiere
ligne, et en silence. D'ou un logging explicite, double d'un garde-fou qui leve
si une cle contenant "accuracy" se presente.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # avant pyplot : aucune fenetre, indispensable en CI

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import (
    DATA_PROCESSED,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
    PROJECT_ROOT,
    REGISTERED_MODEL,
    SPLIT_MANIFEST,
)
from src.training.metrics import confusion_at_threshold

if TYPE_CHECKING:  # pragma: no cover - uniquement pour le typage
    from src.training.train import RunResult, Split

# skops (serialisation par defaut de MLflow 3) refuse d'instancier des types
# qu'on ne lui a pas declares : c'est sa raison d'etre, il ne veut pas executer
# du code arbitraire au chargement. On declare donc explicitement ce qu'on
# approuve, plutot que de retomber sur cloudpickle qui, lui, ne verifie rien.
SKOPS_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]

# Regle de CLAUDE.md, appliquee a l'execution et pas seulement en commentaire.
FORBIDDEN_METRIC_SUBSTRINGS = ("accuracy",)


class TrackingError(RuntimeError):
    """Enregistrement MLflow impossible."""


@dataclass(frozen=True)
class TrackedRun:
    model_kind: str
    run_id: str
    version: str | None


# ---------------------------------------------------------------------------
# Garde-fous et metadonnees
# ---------------------------------------------------------------------------


def assert_no_forbidden_metrics(metrics: dict[str, float]) -> None:
    """Refuse d'enregistrer une metrique interdite.

    Sur un dataset a 0,17 % de positifs, predire "jamais de fraude" donne
    99,83 % d'accuracy en n'attrapant rien. La regle vaut a l'entrainement comme
    a l'enregistrement : une metrique trompeuse est encore plus nuisible une
    fois affichee dans une interface de comparaison.
    """
    offenders = sorted(
        key
        for key in metrics
        if any(token in key.lower() for token in FORBIDDEN_METRIC_SUBSTRINGS)
    )
    if offenders:
        raise TrackingError(
            f"Metriques interdites : {offenders}. Ce projet ne rapporte jamais "
            "d'accuracy (voir CLAUDE.md)."
        )


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_tags() -> dict[str, str]:
    """Commit, branche, et surtout : l'arbre de travail etait-il modifie ?

    Un run lance depuis un arbre sale n'est PAS reproductible a partir du seul
    SHA, puisque le code execute ne correspond a aucun commit. Il faut que ca se
    voie dans l'interface, sinon on croira pouvoir rejouer un run qu'on ne
    pourra pas rejouer.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status is None:
        dirty = "unknown"
    else:
        dirty = "true" if status else "false"
    return {
        "git_commit": commit or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": dirty,
    }


def dataset_tags(processed_dir: Path = DATA_PROCESSED) -> dict[str, str]:
    """Relie le run aux octets exacts dont il est issu, via le manifeste."""
    manifest_path = processed_dir / SPLIT_MANIFEST.name
    if not manifest_path.exists():
        return {"dataset_sha256": "unknown"}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    return {
        "dataset_sha256": str(source.get("sha256", "unknown")),
        "dataset_rows": str(source.get("rows", "unknown")),
        "split_strategy": str(manifest.get("strategy", "unknown")),
        "split_seed": str(manifest.get("seed")),
    }


def prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """``pr_auc`` devient ``val_pr_auc``.

    evaluate() renvoie des cles nues pour ignorer la notion de split ; c'est
    ici, et seulement ici, qu'on decide de quel fold il s'agit.
    """
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


# ---------------------------------------------------------------------------
# Artefacts graphiques
# ---------------------------------------------------------------------------


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    baseline: float,
    title: str,
    path: Path,
) -> Path:
    """Courbes precision-rappel val + test, avec la ligne du hasard.

    La ligne du hasard est tracee parce qu'une PR-AUC ne se lit QUE comparee a
    elle : 0,87 est excellent ici et serait mediocre sur l'echelle du ROC.
    """
    figure, axes = plt.subplots(figsize=(7.0, 5.0))
    for (label, (y_true, y_score)), color in zip(
        curves.items(), ("#4C78A8", "#F58518"), strict=False
    ):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        area = average_precision_score(y_true, y_score)
        axes.plot(recall, precision, color=color, lw=2, label=f"{label} (PR-AUC {area:.4f})")

    axes.axhline(
        baseline,
        color="grey",
        ls="--",
        lw=1.2,
        label=f"hasard ({baseline:.5f})",
    )
    axes.set_xlabel("Rappel  (part des fraudes attrapees)")
    axes.set_ylabel("Precision  (part des alertes justes)")
    axes.set_title(title)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.grid(alpha=0.25)
    axes.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def plot_confusion(counts: Any, threshold: float, title: str, path: Path) -> Path:
    """Matrice de confusion au seuil fige.

    La couleur est normalisee PAR LIGNE : avec ~56 800 vrais negatifs contre ~80
    vrais positifs, une echelle absolue rendrait trois cases sur quatre
    invisibles. Les nombres affiches restent les comptes bruts.
    """
    matrix = np.array([[counts.tn, counts.fp], [counts.fn, counts.tp]], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    shaded = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)

    figure, axes = plt.subplots(figsize=(5.4, 4.6))
    axes.imshow(shaded, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            axes.text(
                column,
                row,
                f"{value:,}".replace(",", " "),
                ha="center",
                va="center",
                fontsize=13,
                color="white" if shaded[row, column] > 0.5 else "#1a1a1a",
            )
    axes.set_xticks([0, 1], ["predit normal", "predit fraude"])
    axes.set_yticks([0, 1], ["reel normal", "reel fraude"])
    axes.set_title(f"{title}\nseuil = {threshold:.6f}")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Session et enregistrement
# ---------------------------------------------------------------------------


def start_session(
    experiment: str = EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> MlflowClient:
    """Fixe l'URI de tracking et l'experience, puis renvoie un client."""
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment(experiment)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def log_run(
    result: RunResult,
    splits: dict[str, Split],
    *,
    client: MlflowClient,
    processed_dir: Path = DATA_PROCESSED,
    register: bool = True,
) -> TrackedRun:
    """Journalise un run complet : params, metriques, tags, artefacts, modele."""
    val_metrics = prefixed(result.val, "val")
    test_metrics = prefixed(result.test, "test")
    assert_no_forbidden_metrics({**val_metrics, **test_metrics})

    train, val, test = splits["train"], splits["val"], splits["test"]
    val_scores = result.model.predict_proba(val.X)[:, 1]
    test_scores = result.model.predict_proba(test.X)[:, 1]

    with mlflow.start_run(run_name=result.model_kind) as active:
        # --- parametres : les ENTREES, on filtre dessus ---------------------
        mlflow.log_params(
            {
                "model_kind": result.model_kind,
                "seed": result.seed,
                "n_train_rows": result.n_train_rows,
                "n_train_positives": result.n_train_positives,
                "best_iteration": result.best_iteration,
                **{key: value for key, value in result.params.items()},
            }
        )

        # --- metriques : les SORTIES numeriques, on trie dessus -------------
        mlflow.log_metrics({**val_metrics, **test_metrics})
        mlflow.log_metric("fit_seconds", result.fit_seconds)
        if result.frozen_threshold is not None:
            mlflow.log_metric("frozen_threshold", result.frozen_threshold)

        # --- tags : la tracabilite -----------------------------------------
        mlflow.set_tags(
            {
                **git_tags(),
                **dataset_tags(processed_dir),
                "threshold_attainable": str(result.threshold_attainable),
                "threshold_source": "validation",
                "warnings": " | ".join(result.warnings) if result.warnings else "",
            }
        )

        # --- artefacts ------------------------------------------------------
        # Les fichiers sont fabriques dans un dossier temporaire, puis confies a
        # mlflow.log_artifact qui decide seul de leur destination. Tenter de
        # deviner le repertoire d'artefacts depuis artifact_uri est un piege :
        # sous Windows l'URI est "file:C:/..." (un seul deux-points), et tout
        # decoupage naif produit un chemin invalide.
        with tempfile.TemporaryDirectory(prefix="fraud-artifacts-") as tmp:
            staging = Path(tmp)

            pr_path = plot_pr_curves(
                {"validation": (val.y, val_scores), "test": (test.y, test_scores)},
                baseline=result.test["pr_auc_baseline"],
                title=f"Courbe precision-rappel - {result.model_kind}",
                path=staging / "pr_curve.png",
            )
            mlflow.log_artifact(str(pr_path), artifact_path="plots")

            if result.frozen_threshold is not None:
                counts = confusion_at_threshold(
                    test.y, test_scores, result.frozen_threshold
                )
                cm_path = plot_confusion(
                    counts,
                    result.frozen_threshold,
                    title=f"Test au seuil fige - {result.model_kind}",
                    path=staging / "confusion_matrix.png",
                )
                mlflow.log_artifact(str(cm_path), artifact_path="plots")

            metrics_path = staging / "metrics.json"
            metrics_path.write_text(
                json.dumps({"val": result.val, "test": result.test}, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(metrics_path), artifact_path="reports")

            manifest_path = processed_dir / SPLIT_MANIFEST.name
            if manifest_path.exists():
                mlflow.log_artifact(str(manifest_path), artifact_path="reports")

        # --- modele : signature + alias-ready --------------------------------
        sample = train.X.head(5)
        # La signature de SORTIE est inferee de predict_proba COMPLET (2
        # colonnes), parce que c'est exactement ce que pyfunc renverra.
        signature = infer_signature(sample, result.model.predict_proba(sample))
        info = mlflow.sklearn.log_model(
            result.model,
            name="model",
            signature=signature,
            input_example=sample,
            # Sans ceci, le modele recharge renvoie des etiquettes 0/1. Un
            # scoreur de fraude doit renvoyer un SCORE : c'est le service qui
            # applique le seuil, pas le modele.
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            registered_model_name=REGISTERED_MODEL if register else None,
        )

        run_id = active.info.run_id

    version = getattr(info, "registered_model_version", None)
    if register and version is None:
        candidates = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        matching = [v for v in candidates if v.run_id == run_id]
        version = matching[0].version if matching else None

    return TrackedRun(
        model_kind=result.model_kind,
        run_id=run_id,
        version=str(version) if version is not None else None,
    )


def promote_best(
    results: list[RunResult],
    tracked: dict[str, TrackedRun],
    *,
    client: MlflowClient,
    min_pr_auc: float,
) -> TrackedRun:
    """Deplace l'alias @production vers le meilleur run, sous condition.

    La promotion n'est JAMAIS automatique : un run experimental ne doit pas
    devenir la production parce qu'il a fini de tourner. En Phase 4 c'est la CI
    qui deviendra l'autorite de promotion ; ce plancher en est la maquette.
    """
    ranked = sorted(results, key=lambda item: item.test["pr_auc"], reverse=True)
    if not ranked:
        raise TrackingError("Aucun run a promouvoir.")

    best = ranked[0]
    score = best.test["pr_auc"]
    if score < min_pr_auc:
        raise TrackingError(
            f"Promotion refusee : le meilleur run ({best.model_kind}) affiche une "
            f"PR-AUC de test de {score:.4f}, sous le plancher de {min_pr_auc:.4f}.\n"
            "  L'alias @production reste sur la version precedente."
        )

    entry = tracked.get(best.model_kind)
    if entry is None or entry.version is None:
        raise TrackingError(
            f"Le run {best.model_kind} n'a pas de version enregistree : "
            "impossible de poser l'alias."
        )

    client.set_registered_model_alias(
        REGISTERED_MODEL, PRODUCTION_ALIAS, entry.version
    )
    return entry
