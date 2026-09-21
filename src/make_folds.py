"""Genera los folds de validación cruzada y los guarda en data/folds.csv.

Usamos StratifiedKFold sobre TARGET porque la clase positiva (default) es
minoritaria (~8%) y queremos la misma proporción en cada fold.

Los folds se generan UNA sola vez y se versionan en git: todos los modelos del
grupo se validan contra exactamente la misma partición, así los AUC son
comparables entre experimentos.

Uso (desde la raíz del repo):
    python -m src.make_folds
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import FOLDS_PATH, ID_COL, N_FOLDS, SEED, TARGET_COL
from src.data import load_application


def make_folds(train: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas [SK_ID_CURR, fold]."""
    # Ordenamos por ID para que el resultado no dependa del orden del CSV
    train = train[[ID_COL, TARGET_COL]].sort_values(ID_COL).reset_index(drop=True)

    folds = pd.DataFrame({ID_COL: train[ID_COL], "fold": -1})
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(skf.split(train, train[TARGET_COL])):
        folds.loc[val_idx, "fold"] = k

    assert (folds["fold"] >= 0).all(), "Quedaron filas sin fold asignado"
    return folds


def main() -> None:
    train = load_application("train", usecols=[ID_COL, TARGET_COL])
    folds = make_folds(train)
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_PATH, index=False)

    # Chequeo rápido: tamaño y tasa de default por fold
    resumen = folds.merge(train, on=ID_COL).groupby("fold")[TARGET_COL].agg(["size", "mean"])
    print(f"Folds guardados en {FOLDS_PATH}")
    print(resumen.rename(columns={"size": "n", "mean": "tasa_default"}))


if __name__ == "__main__":
    main()
