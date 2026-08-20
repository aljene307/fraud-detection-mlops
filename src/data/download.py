"""Recuperation et validation du dataset ULB "Credit Card Fraud Detection".

    python -m src.data.download

Le telechargement est la partie facile. La partie qui compte est la validation :
un CSV tronque ou corrompu produirait un modele d'apparence parfaitement normale
entraine sur des donnees fausses, et rien ne le signalerait avant la production.
Ce module refuse donc de laisser passer un fichier qui n'est pas exactement le
dataset attendu.

Sortie non nulle en cas d'echec, pour que la CI de la Phase 4 puisse bloquer.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    EXPECTED_COLUMNS,
    EXPECTED_FRAUDS,
    EXPECTED_ROWS,
    EXPECTED_SHA256,
    RAW_CSV,
    TARGET,
)

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"

# On lit par blocs de 1 Mo : le fichier fait ~144 Mo et il est inutile de le
# charger entierement en memoire uniquement pour le hacher.
_HASH_CHUNK = 1024 * 1024


class DatasetError(RuntimeError):
    """Erreur de base pour tout ce qui concerne le dataset brut."""


class DatasetNotFoundError(DatasetError):
    """Le CSV est absent et n'a pas pu etre telecharge."""


class DatasetValidationError(DatasetError):
    """Le CSV est present mais n'est pas le dataset attendu."""


@dataclass(frozen=True)
class DatasetStats:
    """Resume d'un dataset dont la validation a reussi."""

    path: Path
    rows: int
    columns: int
    frauds: int
    fraud_rate: float
    sha256: str
    size_bytes: int
    sha256_matches_reference: bool

    def render(self) -> str:
        bar = "=" * 70
        seal = (
            "conforme a la reference"
            if self.sha256_matches_reference
            else "DIFFERENTE de la reference (acceptee via --allow-hash-mismatch)"
        )
        return "\n".join(
            [
                "",
                bar,
                "  DATASET VALIDE",
                bar,
                f"  Fichier   : {self.path}",
                f"  Taille    : {self.size_bytes / 1024 / 1024:.1f} Mo",
                f"  Lignes    : {self.rows:,}".replace(",", " "),
                f"  Colonnes  : {self.columns}",
                f"  Fraudes   : {self.frauds} ({self.fraud_rate:.4%})",
                f"  SHA-256   : {self.sha256}",
                f"              [{seal}]",
                bar,
                "",
                "  Cette empreinte taguera chaque run MLflow en Phase 1: elle permet",
                "  de prouver plus tard quels octets exacts ont produit un modele.",
                "",
            ]
        )


def sha256_of(path: Path) -> str:
    """Empreinte SHA-256 du fichier, calculee en streaming.

    Un seul octet different produit une empreinte totalement differente : c'est
    ce qui permet de detecter une troncature ou une corruption silencieuse.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _manual_instructions(dest: Path) -> str:
    """Message d'aide affiche quand le CSV est introuvable."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return "\n".join(
        [
            f"Le fichier {dest.name} est introuvable et n'a pas pu etre telecharge.",
            "",
            f"  Emplacement attendu :  {dest}",
            "",
            "  --- Option A : depot manuel (le plus rapide, aucun token requis) ---",
            "",
            f"    1. Ouvrir      {KAGGLE_URL}",
            "    2. Cliquer     'Download' (compte Kaggle gratuit requis)",
            f"    3. Dezipper et placer creditcard.csv dans :  {dest.parent}",
            "    4. Relancer    python -m src.data.download",
            "",
            "  --- Option B : API Kaggle (necessaire pour automatiser la CI) ---",
            "",
            "    1. Kaggle -> Settings -> API -> 'Create New API Token'",
            f"    2. Placer le kaggle.json telecharge dans :  {kaggle_json}",
            "    3. Relancer    python -m src.data.download",
            "",
        ]
    )


def _hash_mismatch_message(digest: str, *, alone: bool) -> str:
    """Message affiche quand l'empreinte ne correspond pas a la reference.

    ``alone=True`` signifie que tous les autres controles sont passes : le
    contenu est conforme mais les octets different, ce qui pointe vers une
    cause tres differente d'une corruption.
    """
    lines = [
        "  - Empreinte SHA-256 differente de la reference.",
        f"      attendue : {EXPECTED_SHA256}",
        f"      obtenue  : {digest}",
    ]
    if alone:
        lines += [
            "",
            "    Tous les autres controles passent : le CONTENU est conforme, seuls",
            "    les OCTETS different. Cas typiques : fichier re-enregistre par un",
            "    tableur, fins de ligne converties, ou export d'une autre source.",
        ]
    lines += [
        "",
        "    Si ce fichier doit devenir la nouvelle reference :",
        "      1. relancer avec --allow-hash-mismatch pour valider le reste",
        f'      2. dans src/config.py :  EXPECTED_SHA256 = "{digest}"',
    ]
    return "\n".join(lines)


def download_from_kaggle(dest_dir: Path) -> Path:
    """Telecharge et dezippe le dataset via l'API Kaggle.

    L'import de ``kaggle`` est fait ICI, et non au niveau du module : le paquet
    cherche ~/.kaggle/kaggle.json des l'import et leve une exception s'il ne le
    trouve pas. Un import en tete de fichier ferait donc planter tout module
    important src.data -- y compris `python -m src.features` -- sur une machine
    sans credentials Kaggle.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # credentials absents, paquet casse, etc.
        raise DatasetNotFoundError(f"API Kaggle indisponible : {exc}") from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=str(dest_dir), unzip=True)
    except Exception as exc:
        raise DatasetNotFoundError(f"Telechargement Kaggle echoue : {exc}") from exc

    csv_path = dest_dir / "creditcard.csv"
    if not csv_path.exists():
        raise DatasetNotFoundError(
            f"Telechargement termine mais {csv_path.name} est absent de {dest_dir}."
        )
    return csv_path


def validate_dataset(path: Path, *, allow_hash_mismatch: bool = False) -> DatasetStats:
    """Valide le CSV brut, ou leve DatasetValidationError avec le detail.

    Les controles sont ordonnes en couches : chacune suppose la precedente
    acquise. On ne peut pas compter les fraudes si la colonne Class n'existe
    pas -- on obtiendrait un KeyError illisible au lieu d'un diagnostic.

    En revanche, a l'interieur de la couche "contenu", on collecte TOUS les
    problemes avant d'echouer : sinon chaque correction couterait une relecture
    complete du fichier pour decouvrir l'erreur suivante.
    """
    # --- Couche 1 : le fichier existe et n'est pas vide ---------------------
    if not path.exists():
        raise DatasetNotFoundError(_manual_instructions(path))

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise DatasetValidationError(
            f"{path} existe mais fait 0 octet.\n"
            "Le telechargement a probablement ete interrompu. Supprime le fichier "
            "et recommence."
        )

    # --- Couche 2 : le fichier se parse en CSV ------------------------------
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DatasetValidationError(
            f"{path} n'a pas pu etre lu comme un CSV.\n"
            f"  Erreur pandas : {exc}\n"
            "  Verifie qu'il s'agit bien du CSV dezippe, et non de l'archive .zip."
        ) from exc

    # --- Couche 3 : la structure de colonnes --------------------------------
    actual_columns = tuple(frame.columns)
    if actual_columns != EXPECTED_COLUMNS:
        detail: list[str] = [
            f"Colonnes inattendues dans {path.name}.",
            f"  Attendu : {len(EXPECTED_COLUMNS)} colonnes",
            f"  Trouve  : {len(actual_columns)} colonnes",
        ]
        if set(actual_columns) == set(EXPECTED_COLUMNS):
            detail.append(
                "  Les noms sont corrects mais l'ORDRE differe. Ce n'est pas le "
                "fichier ULB d'origine."
            )
        else:
            missing = [c for c in EXPECTED_COLUMNS if c not in actual_columns]
            extra = [c for c in actual_columns if c not in EXPECTED_COLUMNS]
            if missing:
                detail.append(f"  Manquantes : {missing}")
            if extra:
                detail.append(f"  En trop    : {extra}")
        raise DatasetValidationError("\n".join(detail))

    # --- Couche 4 : le contenu (on collecte tout avant d'echouer) -----------
    problems: list[str] = []

    n_rows = len(frame)
    if n_rows != EXPECTED_ROWS:
        problems.append(
            f"  - Nombre de lignes de donnees : {n_rows:,} au lieu de "
            f"{EXPECTED_ROWS:,} (hors ligne d'en-tete).".replace(",", " ")
        )

    n_frauds = int((frame[TARGET] == 1).sum())
    if n_frauds != EXPECTED_FRAUDS:
        problems.append(
            f"  - Nombre de fraudes : {n_frauds} au lieu de {EXPECTED_FRAUDS}."
        )

    null_counts = frame.isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        listed = ", ".join(f"{col}={int(n)}" for col, n in columns_with_nulls.items())
        problems.append(f"  - Valeurs manquantes : {listed}")

    # Controle 7 : un label corrompu n'est pas forcement un null.
    unexpected_labels = sorted(set(frame[TARGET].unique()) - {0, 1})
    if unexpected_labels:
        problems.append(
            f"  - La colonne {TARGET} contient des valeurs autres que 0/1 : "
            f"{unexpected_labels}"
        )

    # Controle 8 : un CSV corrompu en son milieu se parse souvent en colonne
    # 'object' remplie de texte, sans produire le moindre null.
    non_numeric = [
        col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])
    ]
    if non_numeric:
        problems.append(
            f"  - Colonnes non numeriques : {non_numeric} "
            "(signe classique d'un fichier corrompu en cours de route)"
        )

    # L'empreinte est le dernier controle collecte, et volontairement pas le
    # premier : elle detecte tout, mais ne dit jamais QUOI. En la placant ici,
    # un fichier tronque produit d'abord le diagnostic lisible, puis l'empreinte
    # en complement -- au lieu d'un simple "les octets different".
    digest = sha256_of(path)
    hash_matches = digest == EXPECTED_SHA256
    if not hash_matches and not allow_hash_mismatch:
        problems.append(_hash_mismatch_message(digest, alone=not problems))

    if problems:
        raise DatasetValidationError(
            f"{path.name} n'est pas le dataset attendu. Problemes detectes :\n"
            + "\n".join(problems)
        )

    if not hash_matches:
        print(
            "[data] ATTENTION : empreinte non conforme, acceptee car "
            "--allow-hash-mismatch est actif.",
            file=sys.stderr,
        )

    return DatasetStats(
        path=path,
        rows=n_rows,
        columns=len(actual_columns),
        frauds=n_frauds,
        fraud_rate=n_frauds / n_rows,
        sha256=digest,
        size_bytes=size_bytes,
        sha256_matches_reference=hash_matches,
    )


def ensure_dataset(
    path: Path = RAW_CSV,
    *,
    allow_download: bool = True,
    allow_hash_mismatch: bool = False,
) -> DatasetStats:
    """Garantit la presence d'un dataset valide, ou leve une DatasetError.

    CSV deja present -> on valide. Sinon -> tentative Kaggle, puis validation.
    Si le telechargement echoue, on laisse volontairement validate_dataset lever
    l'erreur : les instructions manuelles ne sont ainsi ecrites qu'a un seul
    endroit.
    """
    if not path.exists() and allow_download:
        print(f"[data] {path.name} absent - tentative via l'API Kaggle...")
        try:
            download_from_kaggle(path.parent)
        except DatasetError as exc:
            print(f"[data] {exc}", file=sys.stderr)

    return validate_dataset(path, allow_hash_mismatch=allow_hash_mismatch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.download",
        description="Telecharge (si possible) et valide le dataset de fraude ULB.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=RAW_CSV,
        help=f"chemin du CSV brut (defaut : {RAW_CSV})",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="ne pas tenter l'API Kaggle : valider uniquement le fichier local",
    )
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="accepter une empreinte SHA-256 differente de EXPECTED_SHA256",
    )
    args = parser.parse_args(argv)

    args.path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stats = ensure_dataset(
            args.path,
            allow_download=not args.no_download,
            allow_hash_mismatch=args.allow_hash_mismatch,
        )
    except DatasetError as exc:
        print("\n" + "!" * 70, file=sys.stderr)
        print("  ECHEC : DATASET INVALIDE OU ABSENT", file=sys.stderr)
        print("!" * 70, file=sys.stderr)
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    print(stats.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
