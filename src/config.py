"""Configuración compartida del proyecto: rutas, semilla y cantidad de folds.

Todo el grupo usa estos mismos valores para que los resultados sean comparables.
"""
import os
from pathlib import Path

# Raíz del repo (esta carpeta está en src/, así que subimos un nivel)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
# Carpeta con los CSV de Kaggle (no se suben a git). Se usa la primera de esta
# lista que tenga application_train.csv, así cada integrante puede dejar los
# datos donde le quede cómodo sin duplicar 2,5 GB:
#   1. la variable de entorno TP_DATA_DIR, si está definida
#   2. data/raw/ dentro del repo
#   3. Archivos_TP/ al lado del repo
#   4. la carpeta que contiene al repo
_CANDIDATES = [
    Path(os.environ["TP_DATA_DIR"]) if os.environ.get("TP_DATA_DIR") else None,
    DATA_DIR / "raw",
    ROOT.parent / "Archivos_TP",
    ROOT.parent,
]
RAW_DIR = next(
    (d for d in _CANDIDATES if d is not None and (d / "application_train.csv").exists()),
    DATA_DIR / "raw",  # si no están en ningún lado, el error va a apuntar a data/raw/
)
FOLDS_PATH = DATA_DIR / "folds.csv"   # asignación SK_ID_CURR -> fold (sí se sube a git)
SUBMISSIONS_DIR = ROOT / "submissions"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

SEED = 42
N_FOLDS = 5
