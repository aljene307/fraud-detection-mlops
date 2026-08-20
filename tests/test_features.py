"""Tests du pipeline de variables et du decoupage (step 0.D).

Les proprietes verifiees ici sont celles dont la violation ne se voit PAS a
l'oeil nu : une ligne partagee entre train et test, une stratification qui a
cesse de stratifier, une graine ignoree. Un modele entraine sur des splits
casses obtient d'excellents scores -- c'est precisement ce qui rend ces bugs
dangereux.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED, TARGET
from src.data.download import DatasetStats
from src.features import build
from src.features.build import (
    FEAT_HOUR,
    FEAT_LOG_AMOUNT,
    FEATURE_COLUMNS,
    build_splits,
    engineer,
    finalise_columns,
    split_by_time,
    split_stratified,
)
from tests.conftest import make_frame

# 85 / 50 000 = 0.17 %, le taux du vrai dataset. Les deux se divisent exactement
# par 60/20/20, ce qui permet une tolerance serree sur la stratification.
N_ROWS = 50_000
N_FRAUDS = 85

SIZES = {"val_size": 0.2, "test_size": 0.2}


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_frame(n_rows=N_ROWS, n_frauds=N_FRAUDS, seed=1)


@pytest.fixture
def engineered(raw: pd.DataFrame) -> pd.DataFrame:
    return engineer(raw)


# --------------------------------------------------------------------------
# Construction des variables
# --------------------------------------------------------------------------


def test_engineer_replaces_amount_and_keeps_time(
    raw: pd.DataFrame, engineered: pd.DataFrame
) -> None:
    """Amount disparait (remplace par log_amount) mais Time reste : le
    decoupage temporel en a besoin, il ne part qu'apres le split."""
    assert "Amount" not in engineered.columns
    assert "Time" in engineered.columns
    assert FEAT_LOG_AMOUNT in engineered.columns
    assert FEAT_HOUR in engineered.columns
    assert len(engineered) == len(raw)


def test_log_amount_is_log1p_of_amount(
    raw: pd.DataFrame, engineered: pd.DataFrame
) -> None:
    """log1p et non log : Amount vaut 0 pour certaines transactions, et log(0)
    donnerait -inf, que XGBoost accepte mais que la regression logistique ne
    supporte pas."""
    assert np.allclose(engineered[FEAT_LOG_AMOUNT], np.log1p(raw["Amount"]))
    assert np.isfinite(engineered[FEAT_LOG_AMOUNT]).all()


def test_hour_of_day_within_range(engineered: pd.DataFrame) -> None:
    """Un modulo mal place produirait des heures a 47 sur une fenetre de 48 h."""
    assert engineered[FEAT_HOUR].between(0, 23).all()


def test_finalise_columns_drops_time_and_fixes_order(
    engineered: pd.DataFrame,
) -> None:
    """Exigence explicite : 30 variables + Class, sans Time ni Amount.

    Time est un decalage d'horloge absolu : le laisser passer permettrait au
    modele de memoriser la periode de collecte, ce qui ne se generalise pas.
    """
    final = finalise_columns(engineered)

    assert len(FEATURE_COLUMNS) == 30
    assert list(final.columns) == [*FEATURE_COLUMNS, TARGET]
    assert "Time" not in final.columns
    assert "Amount" not in final.columns


# --------------------------------------------------------------------------
# Decoupage stratifie
# --------------------------------------------------------------------------


def test_splits_are_disjoint(engineered: pd.DataFrame) -> None:
    """Une seule ligne presente dans train ET dans test est une fuite : le
    modele est evalue sur une donnee qu'il a vue."""
    train, val, test = split_stratified(engineered, seed=42, **SIZES)
    a, b, c = (set(part.index) for part in (train, val, test))

    assert not a & b
    assert not a & c
    assert not b & c


def test_splits_cover_every_row(engineered: pd.DataFrame) -> None:
    """Rien ne doit disparaitre en silence."""
    train, val, test = split_stratified(engineered, seed=42, **SIZES)

    assert len(train) + len(val) + len(test) == len(engineered)
    union = set(train.index) | set(val.index) | set(test.index)
    assert union == set(engineered.index)


def test_stratification_preserves_fraud_rate(engineered: pd.DataFrame) -> None:
    """Le coeur du step. Avec 492 positifs sur le vrai dataset, une
    stratification cassee ferait osciller les metriques pour des raisons sans
    rapport avec le modele.

    Ce test tomberait notamment si le SECOND appel a train_test_split perdait
    son argument stratify -- une erreur invisible en lecture.
    """
    overall = engineered[TARGET].mean()
    train, val, test = split_stratified(engineered, seed=42, **SIZES)

    for part in (train, val, test):
        assert part[TARGET].mean() == pytest.approx(overall, abs=1e-4)


def test_split_proportions_are_60_20_20(engineered: pd.DataFrame) -> None:
    """Verifie l'arithmetique test_size / holdout, facile a rater."""
    train, val, test = split_stratified(engineered, seed=42, **SIZES)
    n = len(engineered)

    assert len(train) / n == pytest.approx(0.6, abs=0.01)
    assert len(val) / n == pytest.approx(0.2, abs=0.01)
    assert len(test) / n == pytest.approx(0.2, abs=0.01)


def test_same_seed_gives_identical_splits(engineered: pd.DataFrame) -> None:
    """Reproductibilite : sans elle, impossible de comparer deux runs MLflow."""
    first = split_stratified(engineered, seed=42, **SIZES)
    second = split_stratified(engineered, seed=42, **SIZES)

    for part_a, part_b in zip(first, second, strict=True):
        assert set(part_a.index) == set(part_b.index)


def test_different_seed_gives_different_splits(engineered: pd.DataFrame) -> None:
    """Contre-epreuve du test precedent : sans lui, une graine totalement
    ignoree passerait pour une reproductibilite parfaite."""
    train_42, _, _ = split_stratified(engineered, seed=42, **SIZES)
    train_7, _, _ = split_stratified(engineered, seed=7, **SIZES)

    assert set(train_42.index) != set(train_7.index)


# --------------------------------------------------------------------------
# Decoupage temporel
# --------------------------------------------------------------------------


def test_time_split_is_chronological(engineered: pd.DataFrame) -> None:
    """Train sur le passe, test sur le futur : le moindre chevauchement rend
    le decoupage temporel inutile."""
    train, val, test = split_by_time(engineered, **SIZES)

    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_time_split_is_disjoint_and_complete(engineered: pd.DataFrame) -> None:
    train, val, test = split_by_time(engineered, **SIZES)

    assert len(train) + len(val) + len(test) == len(engineered)
    union = set(train.index) | set(val.index) | set(test.index)
    assert union == set(engineered.index)


def test_time_split_is_deterministic(engineered: pd.DataFrame) -> None:
    """Le tri stable (mergesort) garantit un decoupage identique a chaque
    appel, meme quand plusieurs transactions partagent la meme valeur de Time."""
    first = split_by_time(engineered, **SIZES)
    second = split_by_time(engineered, **SIZES)

    for part_a, part_b in zip(first, second, strict=True):
        assert list(part_a.index) == list(part_b.index)


# --------------------------------------------------------------------------
# Garde-fous des parametres
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("val_size", "test_size"),
    [
        (0.6, 0.5),  # somme > 1 : train vide
        (0.5, 0.5),  # somme == 1 : train vide
        (0.0, 0.2),  # borne basse
        (0.2, 1.0),  # borne haute
    ],
)
def test_invalid_sizes_rejected(val_size: float, test_size: float) -> None:
    """Les tailles sont validees AVANT toute lecture de fichier : ce test ne
    touche donc jamais au CSV."""
    with pytest.raises(ValueError):
        build_splits(val_size=val_size, test_size=test_size)


# --------------------------------------------------------------------------
# Bout en bout
# --------------------------------------------------------------------------


@pytest.fixture
def stub_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, DatasetStats]:
    """Remplace ensure_dataset par une doublure pointant sur un CSV synthetique.

    Une *doublure* est une fausse fonction substituee a la vraie pendant le
    test. Sans elle, build_splits lirait le fichier de 144 Mo -- et la suite
    deviendrait inexecutable en CI.
    """
    csv_path = tmp_path / "creditcard.csv"
    make_frame(n_rows=N_ROWS, n_frauds=N_FRAUDS, seed=2).to_csv(csv_path, index=False)

    stats = DatasetStats(
        path=csv_path,
        rows=N_ROWS,
        columns=31,
        frauds=N_FRAUDS,
        fraud_rate=N_FRAUDS / N_ROWS,
        sha256="a" * 64,
        size_bytes=csv_path.stat().st_size,
        sha256_matches_reference=True,
    )
    monkeypatch.setattr(build, "ensure_dataset", lambda **_: stats)
    return csv_path, stats


def test_build_splits_writes_files_and_manifest(
    tmp_path: Path, stub_dataset: tuple[Path, DatasetStats]
) -> None:
    out_dir = tmp_path / "processed"
    manifest = build_splits(out_dir=out_dir, seed=42, **SIZES)

    for name in ("train", "val", "test"):
        assert (out_dir / f"{name}.parquet").exists()

    # Le manifeste doit suivre out_dir. Avant correction il partait toujours
    # dans data/processed/, et decrivait donc un decoupage different de celui
    # des parquets poses a cote de lui.
    assert (out_dir / "split_manifest.json").exists()

    written = json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert written == manifest


def test_build_splits_manifest_content(
    tmp_path: Path, stub_dataset: tuple[Path, DatasetStats]
) -> None:
    manifest = build_splits(out_dir=tmp_path / "processed", seed=42, **SIZES)

    assert manifest["strategy"] == "stratified"
    assert manifest["seed"] == 42
    assert manifest["source"]["sha256"] == "a" * 64
    assert manifest["features"] == list(FEATURE_COLUMNS)
    assert manifest["target"] == TARGET

    total = sum(split["rows"] for split in manifest["splits"].values())
    assert total == N_ROWS

    for split in manifest["splits"].values():
        assert split["fraud_rate"] == pytest.approx(N_FRAUDS / N_ROWS, abs=1e-4)


def test_build_splits_output_columns(
    tmp_path: Path, stub_dataset: tuple[Path, DatasetStats]
) -> None:
    """Les parquets relus doivent porter exactement 30 variables + Class."""
    out_dir = tmp_path / "processed"
    build_splits(out_dir=out_dir, seed=42, **SIZES)

    for name in ("train", "val", "test"):
        frame = pd.read_parquet(out_dir / f"{name}.parquet")
        assert list(frame.columns) == [*FEATURE_COLUMNS, TARGET]
        assert "Time" not in frame.columns
        assert "Amount" not in frame.columns


def test_build_splits_parquets_are_disjoint(
    tmp_path: Path, stub_dataset: tuple[Path, DatasetStats]
) -> None:
    """Verification de la propriete de fuite sur les fichiers reellement
    ecrits, et non plus seulement sur les DataFrames en memoire."""
    out_dir = tmp_path / "processed"
    build_splits(out_dir=out_dir, seed=42, **SIZES)

    frames = {
        name: pd.read_parquet(out_dir / f"{name}.parquet")
        for name in ("train", "val", "test")
    }
    # index=False a l'ecriture : on compare les lignes elles-memes.
    keys = {
        name: set(map(tuple, frame.to_numpy())) for name, frame in frames.items()
    }
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]


def test_time_strategy_records_no_seed(
    tmp_path: Path, stub_dataset: tuple[Path, DatasetStats]
) -> None:
    """En mode temporel aucun alea n'intervient : ecrire 42 dans le manifeste
    laisserait croire a une graine qui ne fait rien."""
    manifest = build_splits(out_dir=tmp_path / "processed", strategy="time", **SIZES)

    assert manifest["strategy"] == "time"
    assert manifest["seed"] is None


def test_tests_never_touch_the_real_data_dir(
    tmp_path: Path, stub_dataset: tuple[Path, DatasetStats]
) -> None:
    """Filet de securite : la suite doit etre incapable d'ecraser tes vrais
    splits, meme si quelqu'un oublie un jour de passer out_dir."""
    real_manifest = DATA_PROCESSED / "split_manifest.json"
    before = real_manifest.read_bytes() if real_manifest.exists() else None

    build_splits(out_dir=tmp_path / "processed", seed=42, **SIZES)

    after = real_manifest.read_bytes() if real_manifest.exists() else None
    assert before == after
