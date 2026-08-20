"""Point unique de verite pour les chemins et les identifiants MLflow.

Tout le reste du projet importe d'ici. Deux raisons :

1. Les chemins sont derives de ``__file__``, donc independants du dossier
   courant. ``python -m src.features`` fonctionne alors depuis n'importe ou,
   y compris depuis un conteneur ou un job CI.
2. Le nom du modele enregistre et son alias sont ecrits une seule fois, donc
   l'entrainement (Phase 1) et le serving (Phase 2) ne peuvent pas diverger.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Chemins ---------------------------------------------------------------
# config.py est dans src/, donc parent.parent est la racine du depot.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"

RAW_CSV: Path = DATA_RAW / "creditcard.csv"
SPLIT_MANIFEST: Path = DATA_PROCESSED / "split_manifest.json"
SCALER_PATH: Path = DATA_PROCESSED / "scaler.joblib"

# --- MLflow ----------------------------------------------------------------
# Surchargeable par variable d'environnement : en Phase 5, le pod Kubernetes
# pointera vers un serveur MLflow distant sans qu'une ligne de code ne change.
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri()
)

EXPERIMENT_NAME: str = "fraud-detection"
REGISTERED_MODEL: str = "fraud-detector"

# Les stages du registre (Staging/Production/Archived) sont deprecies depuis
# MLflow 2.9. On utilise un alias : le serving charge
#     models:/fraud-detector@production
PRODUCTION_ALIAS: str = "production"
PRODUCTION_MODEL_URI: str = f"models:/{REGISTERED_MODEL}@{PRODUCTION_ALIAS}"

# --- Donnees ---------------------------------------------------------------
TARGET: str = "Class"

# Valeurs de reference du dataset ULB, utilisees par la validation du step 0.C.
EXPECTED_ROWS: int = 284_807
EXPECTED_FRAUDS: int = 492

# Empreinte du CSV ULB de reference, relevee le 2026-08-20 sur le fichier
# telecharge depuis Kaggle. Toute difference d'octet la fait changer.
#
# Pour adopter volontairement un autre fichier de reference :
#   1. python -m src.data.download --allow-hash-mismatch   (verifie le reste)
#   2. coller ci-dessous l'empreinte affichee par le script
EXPECTED_SHA256: str = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"

# V1..V28 sont les composantes PCA anonymisees ; Time et Amount sont les deux
# seules colonnes brutes. L'ordre compte : on valide la liste exacte.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "Time",
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
    TARGET,
)
