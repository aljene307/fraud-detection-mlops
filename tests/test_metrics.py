"""Tests du module de metriques (step 1.A).

Certains de ces tests sont des DEMONSTRATIONS autant que des verifications :
ils prouvent chiffres a l'appui pourquoi PR-AUC est la metrique principale et
ROC-AUC seulement secondaire. Pour voir les nombres :

    pytest -s -k "flattering or trapezoid"
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve

from src.training.metrics import (
    DEFAULT_MIN_PRECISION,
    MetricError,
    baseline_pr_auc,
    confusion_at_threshold,
    evaluate,
    pr_auc,
    recall_at_precision,
    roc_auc,
)


def _imbalanced(n: int, positives: int, seed: int) -> np.ndarray:
    """Vecteur de labels avec exactement `positives` fraudes."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=positives, replace=False)] = 1
    return labels


# ===========================================================================
# PR-AUC et sa reference
# ===========================================================================


def test_baseline_pr_auc_is_the_positive_rate() -> None:
    """La reference du hasard, sans laquelle une PR-AUC ne veut rien dire."""
    labels = np.array([0] * 997 + [1] * 3)
    assert baseline_pr_auc(labels) == pytest.approx(0.003)


def test_pr_auc_of_perfect_ranking_is_one() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.10, 0.20, 0.30, 0.90, 0.95])
    assert pr_auc(labels, scores) == pytest.approx(1.0)


def test_pr_auc_of_random_scores_falls_back_to_the_base_rate() -> None:
    """Un classement aleatoire vaut le taux de positifs : c'est ce qui donne
    son echelle a la PR-AUC. 0,85 n'impressionne que compare a 0,0017."""
    labels = _imbalanced(n=100_000, positives=170, seed=0)  # 0,17 %
    scores = np.random.default_rng(1).random(labels.size)

    base = baseline_pr_auc(labels)
    assert base == pytest.approx(0.0017, abs=1e-6)
    assert pr_auc(labels, scores) == pytest.approx(base, abs=2e-3)


def test_pr_auc_of_inverted_ranking_is_worse_than_random() -> None:
    """Contre-epreuve : un modele qui classe a l'envers doit tomber sous la
    reference. Sans ce test, une metrique constante passerait le test
    precedent."""
    labels = _imbalanced(n=20_000, positives=100, seed=2)
    scores = 1.0 - labels + np.random.default_rng(3).random(labels.size) * 0.01

    assert pr_auc(labels, scores) < baseline_pr_auc(labels)


# ===========================================================================
# LA demonstration : ROC-AUC flatteuse, PR-AUC honnete
# ===========================================================================


def test_roc_auc_is_flattering_while_pr_auc_is_not() -> None:
    """Le test central du module.

    Construction : 20 fraudes parmi 10 000 transactions. Le modele place bien
    les 20 fraudes en haut du classement -- mais 380 transactions normales
    obtiennent des scores tout aussi eleves.

    Operationnellement c'est un desastre : sur ~400 alertes, 380 sont fausses.
    Le ROC ne le voit pas, parce que ces 380 faux positifs sont divises par
    9 620 vrais negatifs.
    """
    rng = np.random.default_rng(7)
    n_negatives, n_positives = 10_000, 20

    scores_negative = rng.uniform(0.0, 0.5, n_negatives)
    noisy = rng.choice(n_negatives, size=380, replace=False)
    scores_negative[noisy] = rng.uniform(0.8, 1.0, 380)
    scores_positive = rng.uniform(0.8, 1.0, n_positives)

    labels = np.concatenate([np.zeros(n_negatives, int), np.ones(n_positives, int)])
    scores = np.concatenate([scores_negative, scores_positive])

    roc = roc_auc(labels, scores)
    area = pr_auc(labels, scores)
    base = baseline_pr_auc(labels)
    counts = confusion_at_threshold(labels, scores, threshold=0.8)

    print(
        "\n"
        + "=" * 66
        + f"\n  ROC-AUC ............... {roc:.4f}   <- 'excellent modele'"
        + f"\n  PR-AUC ................ {area:.4f}   <- la realite"
        + f"\n  PR-AUC du hasard ...... {base:.4f}"
        + f"\n  Au seuil 0.80 : {counts.tp} vraies fraudes pour {counts.fp} fausses alertes"
        + f"\n  -> precision {counts.precision:.1%}, rappel {counts.recall:.1%}"
        + f"\n  Taux de faux positifs (ce que voit le ROC) : "
        f"{counts.fp / (counts.fp + counts.tn):.2%}"
        + "\n"
        + "=" * 66
    )

    # Le ROC declare un excellent modele...
    assert roc > 0.95
    # ...alors que la precision reelle est catastrophique.
    assert area < 0.15
    assert counts.precision < 0.10
    # Le modele reste tout de meme bien meilleur que le hasard : la PR-AUC le
    # dit sans exagerer, ce que la ROC-AUC est incapable de nuancer.
    assert area > base * 10


# ===========================================================================
# Le piege du trapeze
# ===========================================================================


def test_trapezoid_rule_disagrees_with_average_precision() -> None:
    """Les deux formules ne sont PAS interchangeables.

    Courbe clairsemee construite a la main : 3 fraudes dispersees parmi 15
    transactions, scores strictement decroissants. La precision chute pendant
    qu'on accumule des faux positifs, puis remonte d'un coup a chaque vraie
    fraude rencontree. average_precision_score echantillonne ces remontees ;
    le trapeze coupe a travers les creux en interpolant entre des points de
    fonctionnement qui n'existent pas.

    Le test n'affirme pas un SENS d'ecart -- il n'est pas universel -- mais
    que l'ecart est reel et non negligeable la ou ca compte : sur les courbes
    a peu de positifs, c'est-a-dire notre cas.
    """
    labels = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    scores = np.linspace(1.0, 0.1, labels.size)

    average_precision = pr_auc(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    trapezoid = auc(recall, precision)

    print(
        f"\n  Courbe clairsemee (3 positifs sur 15) :"
        f"\n    average_precision_score : {average_precision:.4f}  <- ce qu'on utilise"
        f"\n    auc(recall, precision)  : {trapezoid:.4f}  <- interpolation invalide"
        f"\n    ecart                   : {trapezoid - average_precision:+.4f}\n"
    )

    assert abs(trapezoid - average_precision) > 0.01


def test_trapezoid_and_average_precision_converge_on_dense_curves() -> None:
    """Complement du test precedent : avec beaucoup de positifs, les deux
    formules se rejoignent presque.

    C'est ce qui rend l'erreur dangereuse -- elle est invisible sur un dataset
    equilibre, et n'apparait que sur un dataset comme le notre.
    """
    labels = _imbalanced(n=5_000, positives=2_000, seed=21)
    rng = np.random.default_rng(22)
    scores = np.where(
        labels == 1,
        rng.uniform(0.3, 1.0, labels.size),
        rng.uniform(0.0, 0.7, labels.size),
    )

    average_precision = pr_auc(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    trapezoid = auc(recall, precision)

    assert abs(trapezoid - average_precision) < 0.005


# ===========================================================================
# recall_at_precision
# ===========================================================================


# Courbe calculee a la main. Scores deja tries par ordre decroissant :
#   k=1 : TP=1 FP=0 -> P=1.000 R=0.25
#   k=2 : TP=2 FP=0 -> P=1.000 R=0.50
#   k=3 : TP=3 FP=0 -> P=1.000 R=0.75
#   k=4 : TP=3 FP=1 -> P=0.750 R=0.75   <- la precision DECROCHE
#   k=5 : TP=4 FP=1 -> P=0.800 R=1.00   <- puis REMONTE
HAND_LABELS = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
HAND_SCORES = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30])


def test_recall_at_precision_hand_computed_target_90() -> None:
    result = recall_at_precision(HAND_LABELS, HAND_SCORES, min_precision=0.90)

    assert result.attainable is True
    assert result.recall == pytest.approx(0.75)
    assert result.threshold == pytest.approx(0.85)
    assert result.achieved_precision == pytest.approx(1.0)


def test_recall_at_precision_scans_past_a_dip_in_precision() -> None:
    """Le test qui compte : la precision decroche a 0,75 avant de remonter a
    0,80. Une implementation qui s'arreterait au premier decrochage renverrait
    un rappel de 0,75 au lieu de 1,00 -- une sous-estimation silencieuse."""
    result = recall_at_precision(HAND_LABELS, HAND_SCORES, min_precision=0.80)

    assert result.recall == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.75)
    assert result.achieved_precision == pytest.approx(0.80)


def test_recall_at_precision_unattainable_returns_none_not_zero() -> None:
    """Aucun pouvoir discriminant : la cible de 90 % est hors de portee.

    threshold DOIT valoir None. Un 0.0 ressemblerait a un vrai seuil et
    serait applique tel quel en Phase 1.
    """
    labels = np.array([1] + [0] * 9)
    scores = np.full(10, 0.5)

    result = recall_at_precision(labels, scores, min_precision=0.90)

    assert result.attainable is False
    assert result.threshold is None
    assert result.recall == 0.0
    assert result.achieved_precision < 0.90


def test_recall_at_precision_with_trivial_target_recovers_everything() -> None:
    result = recall_at_precision(HAND_LABELS, HAND_SCORES, min_precision=0.01)
    assert result.recall == pytest.approx(1.0)


@pytest.mark.parametrize("min_precision", [0.50, 0.70, 0.90, 0.95, 0.99])
def test_returned_threshold_actually_achieves_the_target(min_precision: float) -> None:
    """Test d'aller-retour, et le plus important du module.

    Il attrape le decalage d'indice entre precision[] (longueur n+1) et
    thresholds[] (longueur n), qui ferait renvoyer le mauvais seuil sans lever
    la moindre erreur. Il verifie du meme coup que confusion_at_threshold
    utilise bien la meme convention (score >= seuil) que
    precision_recall_curve.
    """
    labels = _imbalanced(n=5_000, positives=150, seed=13)
    rng = np.random.default_rng(14)
    scores = np.where(
        labels == 1,
        rng.uniform(0.55, 1.0, labels.size),
        rng.uniform(0.0, 0.60, labels.size),
    )

    result = recall_at_precision(labels, scores, min_precision=min_precision)
    assert result.attainable is True

    counts = confusion_at_threshold(labels, scores, result.threshold)
    assert counts.precision >= min_precision - 1e-9
    assert counts.precision == pytest.approx(result.achieved_precision)
    assert counts.recall == pytest.approx(result.recall)


# ===========================================================================
# Matrice de confusion
# ===========================================================================


def test_confusion_at_threshold_hand_computed() -> None:
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.4, 0.8, 0.2, 0.1])

    counts = confusion_at_threshold(labels, scores, threshold=0.5)

    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (1, 1, 1, 2)
    assert counts.precision == pytest.approx(0.5)
    assert counts.recall == pytest.approx(0.5)
    assert counts.total == 5


def test_confusion_counts_always_sum_to_sample_size() -> None:
    labels = _imbalanced(n=3_000, positives=40, seed=15)
    scores = np.random.default_rng(16).random(labels.size)

    for threshold in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert confusion_at_threshold(labels, scores, threshold).total == labels.size


def test_threshold_convention_is_greater_or_equal() -> None:
    """Un score exactement egal au seuil doit declencher une alerte, comme dans
    precision_recall_curve. La convention opposee decalerait tous les seuils."""
    counts = confusion_at_threshold(np.array([1, 0]), np.array([0.5, 0.5]), 0.5)
    assert counts.tp == 1
    assert counts.fp == 1


# ===========================================================================
# evaluate() : le contrat envoye a MLflow
# ===========================================================================


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    labels = _imbalanced(n=5_000, positives=120, seed=17)
    rng = np.random.default_rng(18)
    scores = np.where(
        labels == 1,
        rng.uniform(0.5, 1.0, labels.size),
        rng.uniform(0.0, 0.65, labels.size),
    )
    return labels, scores


def test_evaluate_never_reports_accuracy(scored) -> None:
    """Regle de CLAUDE.md, verifiee par un test plutot que par un commentaire :
    un commentaire ne casse pas la CI."""
    metrics = evaluate(*scored)

    assert not any("accuracy" in key.lower() for key in metrics)
    assert not any(key.lower() in {"acc", "score"} for key in metrics)


def test_evaluate_key_set_is_stable(scored) -> None:
    """MLflow compare des runs entre eux : renommer une cle briserait
    l'historique et le gate CI de la Phase 4."""
    metrics = evaluate(*scored)

    assert set(metrics) == {
        "pr_auc",
        "pr_auc_baseline",
        "pr_auc_lift",
        "roc_auc",
        "recall_at_p90",
        "recall_at_precision_attainable",
        "positives",
        "n_samples",
        "threshold",
        "precision_at_threshold",
        "recall_at_threshold",
        "tp",
        "fp",
        "tn",
        "fn",
    }


def test_evaluate_values_are_all_floats(scored) -> None:
    """MLflow ne stocke que des flottants ; un int passerait mais un bool ou un
    None ferait echouer log_metrics au milieu d'un run."""
    metrics = evaluate(*scored)
    assert all(isinstance(value, float) for value in metrics.values())


def test_evaluate_lift_is_pr_auc_over_baseline(scored) -> None:
    metrics = evaluate(*scored)
    assert metrics["pr_auc_lift"] == pytest.approx(
        metrics["pr_auc"] / metrics["pr_auc_baseline"]
    )
    assert metrics["pr_auc_baseline"] == pytest.approx(120 / 5_000)


def test_evaluate_honours_an_externally_frozen_threshold(scored) -> None:
    """Le cas reel de la Phase 1 : le seuil est choisi sur la VALIDATION puis
    impose au test. Sans ce parametre, on choisirait le seuil sur le test et on
    publierait une metrique gonflee sur ce meme test."""
    labels, scores = scored
    frozen = 0.7

    metrics = evaluate(labels, scores, threshold=frozen)
    expected = confusion_at_threshold(labels, scores, frozen)

    assert metrics["threshold"] == pytest.approx(frozen)
    assert metrics["tp"] == float(expected.tp)
    assert metrics["fp"] == float(expected.fp)


def test_evaluate_target_label_follows_min_precision(scored) -> None:
    metrics = evaluate(*scored, min_precision=0.75)
    assert "recall_at_p75" in metrics
    assert "recall_at_p90" not in metrics


def test_default_min_precision_is_90_percent() -> None:
    assert DEFAULT_MIN_PRECISION == 0.90


# ===========================================================================
# Erreurs : des messages qui disent quoi faire
# ===========================================================================


def test_single_class_raises_with_actionable_message() -> None:
    labels = np.zeros(10, dtype=int)
    scores = np.random.default_rng(19).random(10)

    with pytest.raises(MetricError, match="une seule classe"):
        pr_auc(labels, scores)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(MetricError, match="tailles differentes"):
        pr_auc(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_non_binary_labels_raise() -> None:
    with pytest.raises(MetricError, match="que 0 et 1"):
        pr_auc(np.array([0, 1, 7]), np.array([0.1, 0.2, 0.3]))


def test_non_finite_scores_raise() -> None:
    with pytest.raises(MetricError, match="NaN"):
        pr_auc(np.array([0, 1]), np.array([0.1, np.nan]))


def test_empty_input_raises() -> None:
    with pytest.raises(MetricError, match="vide"):
        pr_auc(np.array([]), np.array([]))


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_min_precision_raises(bad: float) -> None:
    with pytest.raises(MetricError, match="min_precision"):
        recall_at_precision(HAND_LABELS, HAND_SCORES, min_precision=bad)
