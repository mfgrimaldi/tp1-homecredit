"""Armado del archivo de submission para Kaggle (formato SK_ID_CURR, TARGET)."""
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import ID_COL, SUBMISSIONS_DIR, TARGET_COL


def make_submission(ids: pd.Series, preds: np.ndarray, name: str, cv_auc: float | None = None) -> str:
    """Guarda submissions/<fecha>_<name>[_cvXXXX].csv y devuelve la ruta.

    El AUC de CV en el nombre sirve para después comparar contra el score de Kaggle.
    """
    assert len(ids) == len(preds), "ids y predicciones tienen distinto largo"
    assert np.all((preds >= 0) & (preds <= 1)), "las predicciones tienen que ser probabilidades"

    sub = pd.DataFrame({ID_COL: ids.to_numpy(), TARGET_COL: preds})
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = f"_cv{cv_auc:.4f}" if cv_auc is not None else ""
    path = SUBMISSIONS_DIR / f"{stamp}_{name}{suffix}.csv"
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"Submission guardada en {path} ({len(sub)} filas)")
    return str(path)
