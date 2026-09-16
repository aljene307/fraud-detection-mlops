"""Regenere requirements.lock.txt : versions installees, runtime uniquement.

    python scripts/freeze_lock.py

Ce script existe parce qu'un simple `pip freeze > requirements.lock.txt` produit
un lock FAUX a deux titres, et les deux cassent en silence.

**1. Il contient le groupe dev.** Le venv local est installe avec
`pip install -e ".[dev]"`, donc pip freeze y voit pytest, ruff, coverage,
httpx2... L'image Docker s'installe depuis ce lock : elle embarquerait donc
l'outillage de test en production. On calcule ici la CLOTURE RUNTIME (ce que
`pip install .` installerait, sans les extras) et on ne garde que ces noms.

**2. Il contient des paquets Windows-only.** pywin32, tire transitivement par
mlflow, n'existe pas sur Linux, et le build Docker echoue avec :

    ERROR: No matching distribution found for pywin32==312

On lui ajoute le marqueur PEP 508 `; sys_platform == "win32"` : pip l'installe
sous Windows et l'ignore ailleurs, donc UN seul fichier sert aux deux.

Pourquoi les versions INSTALLEES plutot que celles d'une resolution fraiche :
une resolution fraiche interroge PyPI et propose les dernieres versions
compatibles -- mesure faite, 27 paquets divergeaient de l'environnement local.
Le lock doit figer ce qui a ETE TESTE, pas ce qui serait disponible aujourd'hui.
On se sert donc de la resolution uniquement pour connaitre les NOMS a garder.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = PROJECT_ROOT / "requirements.lock.txt"

# Paquets absents de PyPI pour Linux, etablis empiriquement en resolvant le lock
# dans un conteneur python:3.13-slim-trixie jusqu'a ce qu'il passe.
WINDOWS_ONLY = frozenset({"pywin32"})
WINDOWS_MARKER = '; sys_platform == "win32"'

HEADER = """\
# GENERE PAR scripts/freeze_lock.py -- ne pas editer a la main.
#
# Versions installees localement, restreintes a la cloture RUNTIME (sans le
# groupe dev), avec marqueurs de plateforme pour les paquets Windows-only.
# Un `pip freeze` brut produirait un fichier qui casse la construction Docker.
"""


def normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def project_name() -> str:
    """Le projet lui-meme n'a rien a faire dans le lock.

    Il apparait dans la cloture (pip resout `.` en projet + dependances) mais
    pas dans `pip freeze --exclude-editable`, puisqu'il est installe en
    editable. Le Dockerfile copie src/ directement, il n'y a rien a installer.
    """
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return normalise(pyproject["project"]["name"])


def runtime_closure() -> set[str]:
    """Noms des paquets qu'installerait `pip install .` (sans les extras).

    --ignore-installed force une resolution complete : sans ce drapeau, pip ne
    rapporte que ce qui MANQUE, et tout etant deja installe le rapport serait
    vide.
    """
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--dry-run", "--ignore-installed", "--quiet",
                "--report", str(report_path),
                str(PROJECT_ROOT),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

    return {normalise(item["metadata"]["name"]) for item in report["install"]}


def installed_versions() -> dict[str, str]:
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    versions: dict[str, str] = {}
    for line in frozen.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, version = line.partition("==")
        versions[normalise(name)] = version.split(";")[0].strip()
    return versions


def main() -> int:
    closure = runtime_closure() - {project_name()}
    installed = installed_versions()

    kept = sorted(closure & set(installed))
    dropped = sorted(set(installed) - closure)
    missing = sorted(closure - set(installed))

    lines = [HEADER.rstrip()]
    marked: list[str] = []
    for name in kept:
        entry = f"{name}=={installed[name]}"
        if name in WINDOWS_ONLY:
            entry += WINDOWS_MARKER
            marked.append(name)
        lines.append(entry)

    LOCK_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{LOCK_PATH.name} : {len(kept)} paquets runtime")
    if marked:
        print(f"  marques Windows-only : {', '.join(marked)}")
    if dropped:
        print(f"  ecartes (dev-only, {len(dropped)}) : {', '.join(dropped)}")
    if missing:
        print(f"  ATTENTION, dans la cloture mais non installes : {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
