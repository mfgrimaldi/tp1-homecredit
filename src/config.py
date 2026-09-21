"""Configuración compartida del proyecto: rutas, semilla y cantidad de folds.

Todo el grupo usa estos mismos valores para que los resultados sean comparables.
"""
from pathlib import Path

# Raíz del repo (esta carpeta está en src/, así que subimos un nivel)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # acá van los CSV de Kaggle (no se suben a git)
# Si data/raw/ está vacío, buscamos los CSV en la carpeta que contiene al repo
# (así se pueden dejar los datos donde ya estaban, sin duplicar 2,5 GB).
if not (RAW_DIR / "application_train.csv").exists() and (ROOT.parent / "application_train.csv").exists():
    RAW_DIR = ROOT.parent
FOLDS_PATH = DATA_DIR / "folds.csv"   # asignación SK_ID_CURR -> fold (sí se sube a git)
SUBMISSIONS_DIR = ROOT / "submissions"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

SEED = 42
N_FOLDS = 5
