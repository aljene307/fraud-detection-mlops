"""Fixtures partagees par toute la suite de tests.

Regle absolue de ce projet : **aucun test ne lit le vrai CSV de 144 Mo**, sauf
ceux marques ``requires_data``, qui se sautent d'eux-memes quand le fichier est
absent. Tout le reste travaille sur des donnees synthetiques generees ici, ce
qui rend la suite rapide et executable en CI sur une machine vierge.

Les fichiers produits vont tous dans ``tmp_path``, un dossier temporaire fourni
par pytest et efface apres chaque test : le dossier data/ du projet n'est jamais
touche.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import EXPECTED_COLUMNS, TARGET
from src.data import download

# Petit par defaut : les tests de validation ecrivent un CSV a chaque appel, et
# les compteurs attendus sont de toute facon ramenes a ces valeurs par `pin`.
DEFAULT_ROWS = 500
DEFAULT_FRAUDS = 5

SECONDS_IN_48H = 172_800


def make_frame(
    n_rows: int = DEFAULT_ROWS, n_frauds: int = DEFAULT_FRAUDS, seed: int = 0
) -> pd.DataFrame:
    """Construit un DataFrame ayant exactement la forme du dataset ULB.

    Memes colonnes, meme ordre, meme nature (Time en secondes sur 48 h, Amount
    fortement dissymetrique, Class binaire tres desequilibree). Les valeurs sont
    aleatoires : on teste la mecanique du pipeline, pas le contenu du dataset.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "Time": np.sort(rng.uniform(0, SECONDS_IN_48H, n_rows)),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n_rows)
    data["Amount"] = rng.exponential(80.0, n_rows)

    frame = pd.DataFrame(data)
    frame[TARGET] = 0
    fraud_positions = rng.choice(n_rows, size=n_frauds, replace=False)
    frame.iloc[fraud_positions, frame.columns.get_loc(TARGET)] = 1

    return frame[list(EXPECTED_COLUMNS)]


@pytest.fixture
def frame_factory() -> Callable[..., pd.DataFrame]:
    """Donne acces a make_frame aux tests qui veulent une forme particuliere."""
    return make_frame


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[[pd.DataFrame], Path]:
    """Ecrit un DataFrame en CSV dans le dossier temporaire du test."""

    def _write(frame: pd.DataFrame, name: str = "creditcard.csv") -> Path:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return path

    return _write


@pytest.fixture
def valid_csv(write_csv: Callable[[pd.DataFrame], Path]) -> Path:
    """Un CSV synthetique parfaitement conforme aux attentes de `pin`."""
    return write_csv(make_frame())


@pytest.fixture
def pin(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Aligne les compteurs attendus sur les donnees synthetiques.

    validate_dataset compare a EXPECTED_ROWS (284 807) et EXPECTED_FRAUDS (492) :
    aucun fichier synthetique raisonnable ne peut y correspondre. On abaisse donc
    ces constantes le temps du test, pour eprouver la LOGIQUE de validation et
    non les valeurs du dataset ULB -- celles-ci sont couvertes par le test marque
    ``requires_data``.

    On patche les noms dans src.data.download et non dans src.config : le module
    les a importes dans son propre espace de noms, donc c'est la copie locale qui
    est lue a l'execution. monkeypatch restaure tout a la fin du test.
    """

    def _pin(rows: int = DEFAULT_ROWS, frauds: int = DEFAULT_FRAUDS) -> None:
        monkeypatch.setattr(download, "EXPECTED_ROWS", rows)
        monkeypatch.setattr(download, "EXPECTED_FRAUDS", frauds)

    _pin()
    return _pin
