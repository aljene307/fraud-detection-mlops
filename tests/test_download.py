"""Tests de la validation du dataset brut (step 0.C).

Chaque test isole UN mode de defaillance, pour qu'un echec designe directement
le controle casse. Le dernier test fait exception et verifie volontairement une
propriete transverse : la collecte de tous les problemes avant l'echec.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    EXPECTED_FRAUDS,
    EXPECTED_ROWS,
    EXPECTED_SHA256,
    RAW_CSV,
    TARGET,
)
from src.data import download
from src.data.download import (
    DatasetNotFoundError,
    DatasetValidationError,
    sha256_of,
    validate_dataset,
)
from tests.conftest import DEFAULT_FRAUDS, DEFAULT_ROWS

WRONG_HASH = "0" * 64


# --------------------------------------------------------------------------
# Chemin nominal
# --------------------------------------------------------------------------


def test_valid_dataset_passes(pin, valid_csv: Path) -> None:
    """Un fichier conforme doit passer : une validation trop stricte
    refuserait le vrai dataset et bloquerait tout le projet."""
    stats = validate_dataset(valid_csv, allow_hash_mismatch=True)

    assert stats.rows == DEFAULT_ROWS
    assert stats.frauds == DEFAULT_FRAUDS
    assert stats.columns == 31
    assert stats.fraud_rate == pytest.approx(DEFAULT_FRAUDS / DEFAULT_ROWS)
    assert len(stats.sha256) == 64


def test_sha256_of_matches_one_shot_hash(valid_csv: Path) -> None:
    """La lecture par blocs doit donner exactement le meme resultat qu'un
    hachage en une passe : une erreur de boucle produirait une empreinte
    fausse mais stable, donc indetectable autrement."""
    expected = hashlib.sha256(valid_csv.read_bytes()).hexdigest()
    assert sha256_of(valid_csv) == expected


# --------------------------------------------------------------------------
# Couches 1 et 2 : presence, taille, parsing
# --------------------------------------------------------------------------


def test_missing_file_gives_recovery_instructions(tmp_path: Path) -> None:
    """Le message d'absence est la seule chose qui debloque un utilisateur
    sans credentials Kaggle : il doit garder l'URL, le chemin de depot et les
    deux options."""
    missing = tmp_path / "creditcard.csv"

    with pytest.raises(DatasetNotFoundError) as err:
        validate_dataset(missing)

    message = str(err.value)
    assert "kaggle.com" in message
    assert str(missing.parent) in message
    assert "Option A" in message
    assert "Option B" in message


def test_empty_file_rejected(tmp_path: Path, pin) -> None:
    """Un telechargement interrompu laisse un fichier de 0 octet."""
    path = tmp_path / "creditcard.csv"
    path.write_bytes(b"")

    with pytest.raises(DatasetValidationError, match="0 octet"):
        validate_dataset(path)


def test_unparsable_file_rejected(tmp_path: Path, pin) -> None:
    """Cas courant : l'archive .zip renommee ou non dezippee."""
    path = tmp_path / "creditcard.csv"
    path.write_bytes(b"PK\x03\x04 ceci n'est pas un CSV\x00\xff\xfe")

    with pytest.raises(DatasetValidationError):
        validate_dataset(path, allow_hash_mismatch=True)


# --------------------------------------------------------------------------
# Couche 3 : structure des colonnes
# --------------------------------------------------------------------------


def test_missing_column_rejected(pin, write_csv, frame_factory) -> None:
    """Un fichier venant d'une autre source n'a pas les memes colonnes."""
    path = write_csv(frame_factory().drop(columns=["V5"]))

    with pytest.raises(DatasetValidationError) as err:
        validate_dataset(path, allow_hash_mismatch=True)

    message = str(err.value)
    assert "Manquantes" in message
    assert "V5" in message


def test_extra_column_rejected(pin, write_csv, frame_factory) -> None:
    frame = frame_factory()
    frame["extra"] = 1.0
    path = write_csv(frame)

    with pytest.raises(DatasetValidationError) as err:
        validate_dataset(path, allow_hash_mismatch=True)

    assert "En trop" in str(err.value)
    assert "extra" in str(err.value)


def test_column_order_rejected_with_dedicated_message(
    pin, write_csv, frame_factory
) -> None:
    """Memes noms mais ordre different : sans message dedie, l'utilisateur
    lirait 'aucune colonne manquante' et ne comprendrait pas l'echec."""
    frame = frame_factory()
    columns = list(frame.columns)
    columns[1], columns[2] = columns[2], columns[1]
    path = write_csv(frame[columns])

    with pytest.raises(DatasetValidationError, match="ORDRE"):
        validate_dataset(path, allow_hash_mismatch=True)


# --------------------------------------------------------------------------
# Couche 4 : contenu
# --------------------------------------------------------------------------


def test_wrong_row_count_rejected(pin, write_csv, frame_factory) -> None:
    """Un CSV tronque : le cas qui produirait un modele credible entraine sur
    des donnees incompletes."""
    path = write_csv(frame_factory(n_rows=400, n_frauds=DEFAULT_FRAUDS))

    with pytest.raises(DatasetValidationError) as err:
        validate_dataset(path, allow_hash_mismatch=True)

    message = str(err.value)
    assert "lignes de donnees" in message
    assert "400" in message


def test_wrong_fraud_count_rejected(pin, write_csv, frame_factory) -> None:
    path = write_csv(frame_factory(n_rows=DEFAULT_ROWS, n_frauds=9))

    with pytest.raises(DatasetValidationError) as err:
        validate_dataset(path, allow_hash_mismatch=True)

    message = str(err.value)
    assert "fraudes" in message
    assert "9" in message


def test_null_rejected_and_column_named(pin, write_csv, frame_factory) -> None:
    frame = frame_factory()
    frame.loc[3, "V7"] = np.nan
    path = write_csv(frame)

    with pytest.raises(DatasetValidationError) as err:
        validate_dataset(path, allow_hash_mismatch=True)

    message = str(err.value)
    assert "Valeurs manquantes" in message
    assert "V7" in message


def test_non_numeric_column_rejected(pin, write_csv, frame_factory) -> None:
    """Controle 8. Une corruption en milieu de fichier se parse en colonne
    'object' remplie de texte, SANS produire le moindre null : aucun autre
    controle ne l'attrape."""
    frame = frame_factory()
    frame["V3"] = frame["V3"].astype(object)
    frame.loc[10, "V3"] = "corrompu"
    path = write_csv(frame)

    with pytest.raises(DatasetValidationError) as err:
        validate_dataset(path, allow_hash_mismatch=True)

    message = str(err.value)
    assert "non numeriques" in message
    assert "V3" in message


def test_label_outside_zero_one_rejected(pin, write_csv, frame_factory) -> None:
    """Controle 7. Un label corrompu n'est ni un null ni du texte."""
    frame = frame_factory()
    non_fraud_index = frame.index[frame[TARGET] == 0][0]
    frame.loc[non_fraud_index, TARGET] = 7
    path = write_csv(frame)

    with pytest.raises(DatasetValidationError) as err:
        validate_dataset(path, allow_hash_mismatch=True)

    assert "autres que 0/1" in str(err.value)


# --------------------------------------------------------------------------
# Empreinte SHA-256
# --------------------------------------------------------------------------


def test_hash_mismatch_rejected(
    pin, valid_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Le fichier a ete remplace : le message doit donner la nouvelle empreinte
    ET la ligne exacte a coller, sinon l'utilisateur est bloque sans issue."""
    monkeypatch.setattr(download, "EXPECTED_SHA256", WRONG_HASH)

    with pytest.raises(DatasetValidationError) as err:
        validate_dataset(valid_csv)

    message = str(err.value)
    assert "SHA-256" in message
    assert "EXPECTED_SHA256" in message
    assert sha256_of(valid_csv) in message
    # Tous les autres controles passent -> message expliquant que seuls les
    # OCTETS different, pas le contenu.
    assert "CONTENU est conforme" in message


def test_hash_mismatch_accepted_with_flag(
    pin, valid_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Le contournement doit fonctionner, mais rester visible dans le resultat."""
    monkeypatch.setattr(download, "EXPECTED_SHA256", WRONG_HASH)

    stats = validate_dataset(valid_csv, allow_hash_mismatch=True)

    assert stats.sha256_matches_reference is False
    assert "DIFFERENTE de la reference" in stats.render()


def test_hash_match_is_recorded(
    pin, valid_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "EXPECTED_SHA256", sha256_of(valid_csv))

    stats = validate_dataset(valid_csv)

    assert stats.sha256_matches_reference is True
    assert "conforme a la reference" in stats.render()


# --------------------------------------------------------------------------
# Propriete transverse
# --------------------------------------------------------------------------


def test_all_problems_reported_together(pin, write_csv, frame_factory) -> None:
    """LE test de non-regression de la conception : un fichier a la fois
    tronque ET troue doit signaler les DEUX en une seule execution.

    Si quelqu'un remplace un jour la liste `problems` par un `raise` immediat,
    ce test tombe -- et lui seul. Sans lui, chaque correction couterait a
    l'utilisateur une relecture complete de 144 Mo pour decouvrir l'erreur
    suivante.
    """
    frame = frame_factory(n_rows=400, n_frauds=DEFAULT_FRAUDS)
    frame.loc[3, "V7"] = np.nan
    path = write_csv(frame)

    with pytest.raises(DatasetValidationError) as err:
        validate_dataset(path, allow_hash_mismatch=True)

    message = str(err.value)
    assert "lignes de donnees" in message
    assert "Valeurs manquantes" in message


# --------------------------------------------------------------------------
# Vraies donnees : execute en local, saute automatiquement en CI
# --------------------------------------------------------------------------


@pytest.mark.requires_data
@pytest.mark.skipif(not RAW_CSV.exists(), reason=f"{RAW_CSV} absent (normal en CI)")
def test_real_dataset_matches_pinned_constants() -> None:
    """Verifie les VRAIES constantes, que tous les tests ci-dessus contournent.

    C'est le seul test qui lit le fichier de 144 Mo. Il se saute de lui-meme
    quand le CSV est absent, donc la CI reste verte sans lui.
    """
    stats = validate_dataset(RAW_CSV)

    assert stats.rows == EXPECTED_ROWS
    assert stats.frauds == EXPECTED_FRAUDS
    assert stats.sha256 == EXPECTED_SHA256
    assert stats.sha256_matches_reference is True
    assert stats.fraud_rate == pytest.approx(0.001727, abs=1e-6)


@pytest.mark.requires_data
@pytest.mark.skipif(not RAW_CSV.exists(), reason=f"{RAW_CSV} absent (normal en CI)")
def test_real_dataset_columns_exact() -> None:
    header = pd.read_csv(RAW_CSV, nrows=0)
    assert tuple(header.columns) == (
        "Time",
        *(f"V{i}" for i in range(1, 29)),
        "Amount",
        TARGET,
    )
