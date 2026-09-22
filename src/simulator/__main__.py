"""CLI du simulateur de flux : ``python -m src.simulator``.

Volontairement mince : analyse des arguments et affichage. Toute la logique vit
dans replay.py, ou elle est testable sans lancer de service.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import DATA_PROCESSED
from src.simulator.replay import (
    DEFAULT_BASE_URL,
    RunConfig,
    SimulatorError,
    load_stream,
    render_progress,
    render_report,
    run_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.simulator",
        description=(
            "Rejoue le test set en flux vers l'API et mesure debit et latences. "
            "A lancer DANS le reseau Docker : depuis l'exterieur, on mesure le "
            "port forwarding de Docker Desktop (54 ms au lieu de 6 ms)."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=(
            f"racine de l'API (defaut : {DEFAULT_BASE_URL}). Dans le reseau "
            "docker : http://scorer:8000"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("single", "batch"),
        default="batch",
        help=(
            "single : une transaction par requete, mesure la latence qu'un "
            "appelant reel subit. batch : plafond de debit du modele. (defaut : batch)"
        ),
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="debit cible en transactions/s ; 0 = plein regime (defaut : 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="transactions par requete en mode batch (defaut : 500)",
    )
    parser.add_argument(
        "--transactions",
        type=int,
        default=None,
        help="nombre total de transactions a envoyer (defaut : tout le split)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="duree maximale en secondes ; le flux boucle si necessaire",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help=(
            "requetes d'echauffement, envoyees mais EXCLUES des mesures : les "
            "premieres paient des chemins froids (defaut : 3)"
        ),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="ne pas afficher la vue en direct, seulement le rapport final",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = RunConfig(
        base_url=args.base_url,
        mode=args.mode,
        rate=args.rate,
        batch_size=args.batch_size,
        max_transactions=args.transactions,
        duration=args.duration,
        warmup_requests=args.warmup,
        timeout=args.timeout,
    )

    try:
        stream = load_stream(args.processed_dir, args.split)
    except SimulatorError as exc:
        print(f"\nECHEC : {exc}\n", file=sys.stderr)
        return 1

    print(
        f"\nFlux : {len(stream):,} transactions, {stream.n_frauds} fraudes "
        f"({stream.n_frauds / len(stream):.4%}) depuis le split {args.split}".replace(
            ",", " "
        )
    )
    print(f"Cible : {config.url}")

    if config.warmup_requests:
        print(f"Echauffement : {config.warmup_requests} requetes, exclues des mesures")

    # L'avertissement est affiche AVANT la campagne autant qu'apres : mieux vaut
    # qu'il arrete l'utilisateur avant qu'il ne lise des chiffres fausses.
    from src.simulator.replay import OUTSIDE_DOCKER_WARNING, is_outside_docker

    if is_outside_docker(config.base_url):
        print(f"\n{OUTSIDE_DOCKER_WARNING}\n", file=sys.stderr)

    try:
        result = run_campaign(
            stream,
            config,
            on_progress=None if args.quiet else lambda r: print(render_progress(r)),
        )
    except SimulatorError as exc:
        print(f"\nECHEC : {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrompu.\n", file=sys.stderr)
        return 130

    print(render_report(result))

    # L'avertissement complet a deja ete affiche avant la campagne : on se
    # contente ici d'un rappel, pour ne pas le repeter en entier.
    if result.warning:
        print(
            "Rappel : ces chiffres sont contamines par le port forwarding de "
            "Docker Desktop.\n",
            file=sys.stderr,
        )

    # Sortie non nulle si des requetes ont echoue : la CI pourrait s'en servir
    # comme test de fumee apres un deploiement.
    return 1 if result.n_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
