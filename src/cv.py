"""Validación cruzada con los folds fijos del repo.

La idea es que cualquier modelo (logística, árboles, boosting) se evalúe con la
misma función: se pasa un "constructor" que devuelve un estimador de sklearn
sin entrenar (puede ser un Pipeline) y se obtiene:
  * predicciones out-of-fold (OOF) sobre train, para medir AUC y comparar modelos
  * predicciones sobre test, promediando los modelos de cada fold
"""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score


def run_cv(
    build_model: Callable[[], BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    X_test: pd.DataFrame | None = None,
    verbose: bool = True,
) -> dict:
    """Entrena un modelo por fold y devuelve OOF, AUC por fold y predicción de test."""
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    fold_aucs = []

    for k in sorted(folds.unique()):
        tr, va = (folds != k).to_numpy(), (folds == k).to_numpy()

        model = build_model()
        model.fit(X.iloc[tr], y.iloc[tr])

        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y.iloc[va], oof[va])
        fold_aucs.append(auc)
        if verbose:
            print(f"Fold {k}: AUC = {auc:.4f}")

        if X_test is not None:
            # Promedio simple de las predicciones de los modelos de cada fold
            test_pred += model.predict_proba(X_test)[:, 1] / folds.nunique()

    oof_auc = roc_auc_score(y, oof)
    if verbose:
        print(f"AUC OOF total: {oof_auc:.4f} | media por fold: "
              f"{np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    return {"oof": oof, "oof_auc": oof_auc, "fold_aucs": fold_aucs, "test_pred": test_pred}
