"""Construction des variables et decoupage train / validation / test.

Point d'entree : ``python -m src.features`` (voir __main__.py).

Deux principes gouvernent ce module :

1. **Aucune fuite de donnees.** Les transformations appliquees ici sont sans
   etat : elles se calculent ligne par ligne et n'apprennent rien de
   l'ensemble. Toute transformation qui APPREND quelque chose (moyenne,
   ecart-type, encodage) doit vivre dans le Pipeline scikit-learn du modele,
   pour etre ajustee sur le train seul. C'est pourquoi aucun StandardScaler
   n'est ajuste ni sauvegarde ici.

2. **Le split TEST ne sert qu'au chiffre final.** Le seuil de decision et les
   comparaisons de modeles se font sur la VALIDATION. Choisir un seuil sur le
   test puis publier une metrique sur ce meme test gonfle artificiellement le
   resultat et rendrait le gate CI de la Phase 4 mensonger.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED, SPLIT_MANIFEST, TARGET
from src.data.download import ensure_dataset

SplitStrategy = Literal["stratified", "time"]

RAW_TIME = "Time"
RAW_AMOUNT = "Amount"
FEAT_LOG_AMOUNT = "log_amount"
FEAT_HOUR = "hour_of_day"

SECONDS_PER_HOUR = 3600

# Les 30 variables presentees au modele : 28 composantes PCA + 2 derivees.
FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"V{i}" for i in range(1, 29)),
    FEAT_LOG_AMOUNT,
    FEAT_HOUR,
)

SPLIT_FILES: dict[str, str] = {
    "train": "train.parquet",
    "val": "val.parquet",
    "test": "test.parquet",
}


@dataclass(frozen=True)
class SplitSummary:
    """Comptes d'un split, tels qu'ecrits dans le manifeste."""

    rows: int
    frauds: int
    fraud_rate: float
    file: str


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Construit les variables derivees. Sans etat, donc sans fuite possible.

    - ``log_amount`` : Amount est fortement dissymetrique (quelques transactions
      enormes ecrasent tout le reste). log1p compresse cette queue. La
      transformation est monotone, donc INVISIBLE pour XGBoost -- les coupures
      d'arbre ne changent pas -- mais elle aide nettement la regression
      logistique. On retire Amount : log1p etant inversible, rien n'est perdu,
      et garder les deux creerait une paire parfaitement correlee.

    - ``hour_of_day`` : Time est un nombre de secondes depuis la premiere
      transaction, sur environ 48 h. L'heure de la journee porte un signal reel
      (la fraude a un rythme nocturne), contrairement au decalage brut.

    ``Time`` est conserve ici : le decoupage temporel en a besoin. Il est retire
    juste apres le split, par ``finalise_columns``.
    """
    out = frame.copy()
    out[FEAT_LOG_AMOUNT] = np.log1p(out[RAW_AMOUNT])
    out[FEAT_HOUR] = (out[RAW_TIME] // SECONDS_PER_HOUR) % 24
    return out.drop(columns=[RAW_AMOUNT])


def finalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Retire Time et fige l'ordre des colonnes.

    ``Time`` est un decalage d'horloge absolu sur une fenetre de deux jours :
    le conserver laisserait le modele memoriser QUAND les donnees ont ete
    collectees, ce qui ne se generalise a aucune autre periode.
    """
    return frame[[*FEATURE_COLUMNS, TARGET]].copy()


def split_stratified(
    frame: pd.DataFrame, *, seed: int, val_size: float, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decoupage aleatoire stratifie sur Class, en deux temps.

    On isole d'abord le holdout (val + test), puis on le recoupe en deux. Les
    deux appels sont stratifies : avec seulement 492 fraudes, un tirage non
    stratifie ferait varier le nombre de positifs d'un split a l'autre, et les
    metriques bougeraient pour des raisons sans rapport avec le modele.
    """
    holdout = val_size + test_size
    train, rest = train_test_split(
        frame,
        test_size=holdout,
        stratify=frame[TARGET],
        random_state=seed,
        shuffle=True,
    )
    # Part du holdout qui revient au test : 0.2 / 0.4 = 0.5 avec les defauts.
    val, test = train_test_split(
        rest,
        test_size=test_size / holdout,
        stratify=rest[TARGET],
        random_state=seed,
        shuffle=True,
    )
    return train, val, test


def split_by_time(
    frame: pd.DataFrame, *, val_size: float, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decoupage chronologique : on s'entraine sur le passe, on teste sur le futur.

    Plus proche de la realite de production, mais incompatible avec la
    stratification par construction : on ne peut pas imposer les proportions ET
    respecter la chronologie. Le taux de fraude de chaque split est donc subi,
    pas choisi -- le manifeste le rend visible.

    ``kind="mergesort"`` est un tri STABLE : les transactions partageant la meme
    valeur de Time gardent leur ordre d'origine, ce qui rend le decoupage
    reproductible a l'identique.
    """
    ordered = frame.sort_values(RAW_TIME, kind="mergesort")
    n = len(ordered)
    n_train = round(n * (1.0 - val_size - test_size))
    n_val = round(n * val_size)
    return (
        ordered.iloc[:n_train],
        ordered.iloc[n_train : n_train + n_val],
        ordered.iloc[n_train + n_val :],
    )


def summarise(frame: pd.DataFrame, filename: str) -> SplitSummary:
    rows = len(frame)
    frauds = int((frame[TARGET] == 1).sum())
    return SplitSummary(
        rows=rows,
        frauds=frauds,
        fraud_rate=frauds / rows if rows else 0.0,
        file=filename,
    )


def build_splits(
    *,
    strategy: SplitStrategy = "stratified",
    seed: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
    out_dir: Path = DATA_PROCESSED,
    allow_hash_mismatch: bool = False,
) -> dict:
    """Valide la source, construit les variables, decoupe, ecrit, et retourne le manifeste."""
    if not 0.0 < val_size < 1.0 or not 0.0 < test_size < 1.0:
        raise ValueError("val_size et test_size doivent etre strictement entre 0 et 1.")
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size + test_size = {val_size + test_size:.2f} ne laisse aucune "
            "donnee pour l'entrainement."
        )

    # On ne decoupe jamais un fichier non valide : le SHA-256 de la source
    # remonte dans le manifeste et taguera les runs MLflow en Phase 1.
    stats = ensure_dataset(allow_hash_mismatch=allow_hash_mismatch)

    # Seconde lecture du CSV : validate_dataset ne conserve pas le DataFrame,
    # volontairement (sa signature reste "chemin -> statistiques"). Le cout est
    # de quelques secondes, une seule fois.
    raw = pd.read_csv(stats.path)
    engineered = engineer(raw)

    if strategy == "stratified":
        train, val, test = split_stratified(
            engineered, seed=seed, val_size=val_size, test_size=test_size
        )
    elif strategy == "time":
        train, val, test = split_by_time(
            engineered, val_size=val_size, test_size=test_size
        )
    else:  # pragma: no cover - argparse restreint deja les valeurs
        raise ValueError(f"Strategie inconnue : {strategy!r}")

    frames = {
        "train": finalise_columns(train),
        "val": finalise_columns(val),
        "test": finalise_columns(test),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(out_dir / SPLIT_FILES[name], index=False)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "strategy": strategy,
        "seed": seed if strategy == "stratified" else None,
        "val_size": val_size,
        "test_size": test_size,
        "source": {
            "path": str(stats.path),
            "sha256": stats.sha256,
            "rows": stats.rows,
            "frauds": stats.frauds,
        },
        "target": TARGET,
        "features": list(FEATURE_COLUMNS),
        "splits": {
            name: asdict(summarise(frame, SPLIT_FILES[name]))
            for name, frame in frames.items()
        },
    }

    SPLIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def render_manifest(manifest: dict, out_dir: Path) -> str:
    """Rend le manifeste sous forme de tableau lisible en console."""
    bar = "=" * 70
    seed = manifest["seed"]
    seed_label = f", graine : {seed}" if seed is not None else " (pas de graine)"

    lines = [
        "",
        bar,
        f"  SPLITS ECRITS  (strategie : {manifest['strategy']}{seed_label})",
        bar,
        f"  {'split':<8}{'lignes':>12}{'fraudes':>10}{'taux':>12}",
        "  " + "-" * 42,
    ]
    total_rows = 0
    total_frauds = 0
    for name in ("train", "val", "test"):
        split = manifest["splits"][name]
        total_rows += split["rows"]
        total_frauds += split["frauds"]
        rows_fmt = f"{split['rows']:,}".replace(",", " ")
        lines.append(
            f"  {name:<8}{rows_fmt:>12}{split['frauds']:>10}"
            f"{split['fraud_rate']:>11.4%}"
        )
    total_rows_fmt = f"{total_rows:,}".replace(",", " ")
    total_line = (
        f"  {'total':<8}{total_rows_fmt:>12}{total_frauds:>10}"
        f"{total_frauds / total_rows:>11.4%}"
    )
    lines += [
        "  " + "-" * 42,
        total_line,
        "",
        f"  Sortie    : {out_dir}",
        f"  Manifeste : {SPLIT_MANIFEST.name}",
        f"  Source    : SHA-256 {manifest['source']['sha256'][:16]}...",
        "",
        "  Rappel : le split TEST ne sert qu'au chiffre final. Le seuil de",
        "  decision se choisit sur la VALIDATION.",
        bar,
        "",
    ]
    return "\n".join(lines)
