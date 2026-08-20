"""CLI du pipeline de variables : ``python -m src.features``.

Ce fichier reste volontairement mince : il ne fait qu'analyser les arguments et
afficher le resultat. Toute la logique vit dans build.py, ou elle est testable
sans passer par une ligne de commande.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import DATA_PROCESSED
from src.data.download import DatasetError
from src.features.build import build_splits, render_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.features",
        description=(
            "Construit les variables et ecrit les splits train/val/test "
            "dans data/processed/."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=("stratified", "time"),
        default="stratified",
        help=(
            "stratified (defaut) : tirage aleatoire conservant ~0.17%% de fraude "
            "dans chaque split. time : decoupage chronologique sur Time."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="graine aleatoire, ignoree par --strategy=time (defaut : 42)",
    )
    parser.add_argument(
        "--val-size", type=float, default=0.2, help="part de validation (defaut : 0.2)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="part de test (defaut : 0.2)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_PROCESSED,
        help=f"dossier de sortie (defaut : {DATA_PROCESSED})",
    )
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="accepter une source dont le SHA-256 differe de la reference",
    )
    args = parser.parse_args(argv)

    try:
        manifest = build_splits(
            strategy=args.strategy,
            seed=args.seed,
            val_size=args.val_size,
            test_size=args.test_size,
            out_dir=args.out_dir,
            allow_hash_mismatch=args.allow_hash_mismatch,
        )
    except (DatasetError, ValueError) as exc:
        print("\n" + "!" * 70, file=sys.stderr)
        print("  ECHEC : DECOUPAGE IMPOSSIBLE", file=sys.stderr)
        print("!" * 70, file=sys.stderr)
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    print(render_manifest(manifest, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
