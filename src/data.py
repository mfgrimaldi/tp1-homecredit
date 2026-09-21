"""Lectura de datos crudos y de los folds."""
import pandas as pd

from src.config import FOLDS_PATH, ID_COL, RAW_DIR


def load_application(split: str = "train", **read_csv_kwargs) -> pd.DataFrame:
    """Lee application_train.csv o application_test.csv desde data/raw/."""
    if split not in ("train", "test"):
        raise ValueError("split tiene que ser 'train' o 'test'")
    path = RAW_DIR / f"application_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No encuentro {path}. Bajá los datos de Kaggle "
            "(competencia home-credit-default-risk) y dejalos en data/raw/."
        )
    return pd.read_csv(path, **read_csv_kwargs)


def load_folds() -> pd.DataFrame:
    """Lee la asignación SK_ID_CURR -> fold generada por src/make_folds.py."""
    if not FOLDS_PATH.exists():
        raise FileNotFoundError(f"No existe {FOLDS_PATH}. Corré: python -m src.make_folds")
    return pd.read_csv(FOLDS_PATH)


def attach_folds(train: pd.DataFrame) -> pd.Series:
    """Devuelve el fold de cada fila de train, alineado con su índice."""
    folds = load_folds().set_index(ID_COL)["fold"]
    fold = train[ID_COL].map(folds)
    if fold.isna().any():
        raise ValueError("Hay filas de train sin fold: ¿cambió application_train.csv?")
    return fold.astype(int)
