"""Metriques adaptees a un desequilibre extreme (0,17 % de positifs).

Ce module est ecrit AVANT tout entrainement, deliberement : on definit comment
on mesure avant de chercher a optimiser, sinon on optimise ce que l'on regarde.

**L'accuracy est volontairement absente de ce module.** Sur ce dataset, predire
"jamais de fraude" donne 99,83 % d'accuracy en n'attrapant aucune fraude. La
regle est verifiee par un test (test_evaluate_never_reports_accuracy) plutot que
par un commentaire, parce qu'un commentaire ne casse pas la CI.

Metrique principale : **PR-AUC** (aire sous la courbe precision-rappel).
Contrairement au ROC, la courbe PR ne fait intervenir aucun vrai negatif, ni en
abscisse ni en ordonnee. Sur 284 315 transactions normales, le taux de faux
positifs du ROC (FP / (FP + TN)) est anesthesie par ce denominateur enorme : un
modele peut afficher 0,98 de ROC-AUC alors que 96 % de ses alertes sont fausses.
La precision, elle, divise par le nombre d'alertes emises, et dit la verite.

Metrique operationnelle : **rappel a precision fixee**. "Si je n'agis que sur
des alertes sures a 90 %, quelle part de la fraude est-ce que j'attrape ?"
C'est le nombre qu'une equipe fraude lit reellement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

DEFAULT_MIN_PRECISION = 0.90


class MetricError(ValueError):
    """Entree invalide pour un calcul de metrique."""


@dataclass(frozen=True)
class RecallAtPrecision:
    """Resultat d'une recherche de rappel sous contrainte de precision.

    ``threshold`` vaut None quand la cible est inatteignable. On ne renvoie
    JAMAIS 0.0 dans ce cas : un zero ressemblerait a un vrai point de
    fonctionnement, alors qu'il signifie "aucun seuil ne permet cela".
    """

    target_precision: float
    recall: float
    threshold: float | None
    achieved_precision: float
    attainable: bool


@dataclass(frozen=True)
class ConfusionCounts:
    """Comptes a un seuil donne, avec la convention ``score >= seuil``."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        """Part des alertes emises qui etaient de vraies fraudes."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        """Part des vraies fraudes qui ont ete attrapees."""
        actual_positives = self.tp + self.fn
        return self.tp / actual_positives if actual_positives else 0.0


# ---------------------------------------------------------------------------
# Validation des entrees
# ---------------------------------------------------------------------------


def _as_labels(y_true) -> np.ndarray:
    labels = np.asarray(y_true)
    if labels.ndim != 1:
        raise MetricError(f"y_true doit etre unidimensionnel, recu {labels.ndim}D.")
    if labels.size == 0:
        raise MetricError("y_true est vide.")
    distinct = set(np.unique(labels).tolist())
    if not distinct <= {0, 1}:
        raise MetricError(
            f"y_true ne doit contenir que 0 et 1 ; valeurs trouvees : {sorted(distinct)}."
        )
    return labels.astype(int)


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    labels = _as_labels(y_true)
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim != 1:
        raise MetricError(f"y_score doit etre unidimensionnel, recu {scores.ndim}D.")
    if labels.shape != scores.shape:
        raise MetricError(
            f"y_true et y_score ont des tailles differentes : "
            f"{labels.shape} contre {scores.shape}."
        )
    if not np.isfinite(scores).all():
        raise MetricError("y_score contient des NaN ou des valeurs infinies.")
    return labels, scores


def _require_both_classes(labels: np.ndarray) -> None:
    distinct = set(np.unique(labels).tolist())
    if distinct != {0, 1}:
        raise MetricError(
            f"y_true ne contient qu'une seule classe ({sorted(distinct)}) : la "
            "PR-AUC et le rappel ne sont pas definis. Sur ce dataset c'est le "
            "symptome d'un split non stratifie ou trop petit."
        )


# ---------------------------------------------------------------------------
# Metriques
# ---------------------------------------------------------------------------


def baseline_pr_auc(y_true) -> float:
    """PR-AUC d'un classement aleatoire : le taux de positifs (~0,0017 ici).

    C'est la reference sans laquelle une PR-AUC ne veut rien dire. 0,85 est
    excellent pour une PR-AUC (500x le hasard) et mediocre pour une ROC-AUC :
    les deux metriques n'ont pas la meme echelle, et les confondre est une
    erreur classique.
    """
    return float(_as_labels(y_true).mean())


def pr_auc(y_true, y_score) -> float:
    """Aire sous la courbe precision-rappel, via ``average_precision_score``.

    **Pas** ``sklearn.metrics.auc(recall, precision)``. Cette derniere relie les
    points de la courbe par des trapezes, donc suppose des points de
    fonctionnement intermediaires qui n'existent pas. L'interpolation lineaire
    n'est pas valide dans l'espace precision-rappel (Davis & Goadrich, 2006) :
    la relation entre precision et rappel le long d'un segment n'est pas
    lineaire, et l'aire obtenue n'estime rien de defini.

    ``average_precision_score`` calcule AP = somme des (Rn - Rn-1) * Pn, une
    somme de rectangles sans aucune interpolation. C'est l'estimateur standard,
    celui que rapporte la litterature sous le nom "PR-AUC" ou "AP".

    Le SENS de l'ecart entre les deux n'est pas universel : mesure sur des
    courbes clairsemees (peu de positifs), le trapeze passe sous l'AP, parce
    que la precision remonte brutalement a chaque vraie fraude rencontree et
    que l'AP echantillonne justement ces remontees tandis que le trapeze coupe
    a travers les creux. Voir test_trapezoid_rule_disagrees_with_average_precision.
    """
    labels, scores = _as_arrays(y_true, y_score)
    _require_both_classes(labels)
    return float(average_precision_score(labels, scores))


def roc_auc(y_true, y_score) -> float:
    """ROC-AUC, gardee en metrique SECONDAIRE uniquement.

    Utile pour comparer avec la litterature, mais trompeuse ici : voir la
    docstring du module. Ne jamais l'utiliser comme critere de decision.
    """
    labels, scores = _as_arrays(y_true, y_score)
    _require_both_classes(labels)
    return float(roc_auc_score(labels, scores))


def recall_at_precision(
    y_true, y_score, min_precision: float = DEFAULT_MIN_PRECISION
) -> RecallAtPrecision:
    """Meilleur rappel atteignable sans descendre sous ``min_precision``.

    Renvoie aussi le seuil qui realise ce point : c'est lui que l'on fige sur la
    validation avant de l'appliquer au test.
    """
    if not 0.0 < min_precision <= 1.0:
        raise MetricError(
            f"min_precision doit etre dans ]0, 1], recu {min_precision}."
        )
    labels, scores = _as_arrays(y_true, y_score)
    _require_both_classes(labels)

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # precision et recall comptent len(thresholds) + 1 elements : sklearn ajoute
    # un point sentinelle final (precision=1, recall=0) qui ne correspond a AUCUN
    # seuil. On le retire pour que l'indice i designe toujours thresholds[i].
    # Sans cette coupe, on lit soit hors des bornes, soit -- bien pire -- le
    # mauvais seuil, silencieusement.
    precision = precision[:-1]
    recall = recall[:-1]

    eligible = precision >= min_precision
    if not eligible.any():
        return RecallAtPrecision(
            target_precision=min_precision,
            recall=0.0,
            threshold=None,
            achieved_precision=float(precision.max()) if precision.size else 0.0,
            attainable=False,
        )

    # La precision n'est PAS monotone le long de la courbe : aux seuils eleves
    # elle oscille, ses denominateurs etant minuscules. On balaie donc tous les
    # points eligibles au lieu de s'arreter au premier decrochage.
    best = int(np.argmax(np.where(eligible, recall, -np.inf)))

    return RecallAtPrecision(
        target_precision=min_precision,
        recall=float(recall[best]),
        threshold=float(thresholds[best]),
        achieved_precision=float(precision[best]),
        attainable=True,
    )


def confusion_at_threshold(y_true, y_score, threshold: float) -> ConfusionCounts:
    """Compte TP/FP/TN/FN a un seuil donne.

    Convention ``score >= seuil``, identique a celle de
    ``precision_recall_curve``. Une divergence ici rendrait le seuil renvoye par
    ``recall_at_precision`` inapplicable -- d'ou un test d'aller-retour dedie.
    """
    labels, scores = _as_arrays(y_true, y_score)
    flagged = scores >= float(threshold)
    is_fraud = labels == 1
    return ConfusionCounts(
        tp=int(np.count_nonzero(flagged & is_fraud)),
        fp=int(np.count_nonzero(flagged & ~is_fraud)),
        tn=int(np.count_nonzero(~flagged & ~is_fraud)),
        fn=int(np.count_nonzero(~flagged & is_fraud)),
    )


def evaluate(
    y_true,
    y_score,
    *,
    threshold: float | None = None,
    min_precision: float = DEFAULT_MIN_PRECISION,
) -> dict[str, float]:
    """Assemble le dictionnaire de metriques envoye a MLflow.

    Les cles ne sont pas prefixees : ``train.py`` ajoutera ``val_`` ou ``test_``.
    Ainsi ce module ignore tout de la notion de split.

    ``threshold=None`` fait retomber sur le seuil issu de la contrainte de
    precision. En Phase 1, on passera explicitement le seuil FIGE SUR LA
    VALIDATION lors de l'evaluation du test : choisir un seuil sur le test puis
    publier une metrique sur ce meme test gonfle le resultat.

    Toutes les valeurs sont des float, y compris les comptes : MLflow ne stocke
    que des nombres flottants.
    """
    labels, scores = _as_arrays(y_true, y_score)
    _require_both_classes(labels)

    baseline = baseline_pr_auc(labels)
    area = pr_auc(labels, scores)
    found = recall_at_precision(labels, scores, min_precision=min_precision)

    target_label = f"recall_at_p{round(min_precision * 100)}"
    metrics: dict[str, float] = {
        "pr_auc": area,
        "pr_auc_baseline": baseline,
        "pr_auc_lift": area / baseline if baseline else float("nan"),
        "roc_auc": roc_auc(labels, scores),
        target_label: found.recall,
        "recall_at_precision_attainable": float(found.attainable),
        "positives": float(labels.sum()),
        "n_samples": float(labels.size),
    }

    chosen = threshold if threshold is not None else found.threshold
    if chosen is not None:
        counts = confusion_at_threshold(labels, scores, chosen)
        metrics.update(
            {
                "threshold": float(chosen),
                "precision_at_threshold": counts.precision,
                "recall_at_threshold": counts.recall,
                "tp": float(counts.tp),
                "fp": float(counts.fp),
                "tn": float(counts.tn),
                "fn": float(counts.fn),
            }
        )
    return metrics
