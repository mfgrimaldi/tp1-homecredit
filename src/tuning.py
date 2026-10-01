"""Optimización de hiperparámetros de LightGBM con Optuna sobre los folds fijos.

Cada combinación de hiperparámetros se evalúa con validación cruzada de 5 folds
(los mismos de data/folds.csv). Para cada una se usa early stopping: se agregan
árboles mientras el AUC de validación siga mejorando, así la cantidad de árboles
también queda optimizada y no hay que buscarla aparte.

Optuna (algoritmo TPE) propone cada combinación nueva mirando cuáles dieron
mejor AUC hasta el momento, por lo que necesita menos pruebas que una grilla.
"""
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.config import SEED
from src.models import CAT_DTYPES

# Learning rate alto durante la búsqueda para que cada prueba sea rápida;
# el modelo final se reentrena con uno más bajo (ver final_params)
SEARCH_LR = 0.05


def _to_category(X: pd.DataFrame) -> pd.DataFrame:
    """Columnas de texto -> 'category' (categorías tomadas de todo X; no usa el target)."""
    X = X.copy()
    for c in X.select_dtypes(include=CAT_DTYPES).columns:
        X[c] = X[c].astype("category")
    return X


def _fold_indices(folds: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convierte la columna de folds en pares (índices de train, índices de validación)."""
    f = folds.to_numpy()
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in sorted(np.unique(f))]


def _suggest(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda: qué valores puede probar cada hiperparámetro."""
    return {
        # Complejidad de cada árbol
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        # Aleatoriedad: filas y columnas que ve cada árbol (reduce sobreajuste)
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.8),
        # Regularización: penaliza hojas con valores extremos
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    }


def tune_lgbm(X: pd.DataFrame, y: pd.Series, folds: pd.Series, n_trials: int = 30) -> optuna.Study:
    """Corre la búsqueda y devuelve el estudio de Optuna (con todas las pruebas)."""
    dtrain = lgb.Dataset(_to_category(X), label=y, free_raw_data=False)
    cv_folds = _fold_indices(folds)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary", "metric": "auc", "learning_rate": SEARCH_LR,
            "subsample_freq": 1, "seed": SEED, "verbose": -1, "n_jobs": -1,
            **_suggest(trial),
        }
        res = lgb.cv(params, dtrain, num_boost_round=5000, folds=cv_folds,
                     callbacks=[lgb.early_stopping(100, verbose=False)])
        # Guardamos cuántos árboles hicieron falta para usarlo en el modelo final
        trial.set_user_attr("n_estimators", len(res["valid auc-mean"]))
        return res["valid auc-mean"][-1]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def final_params(study: optuna.Study, learning_rate: float = 0.02) -> dict:
    """Mejores hiperparámetros para el modelo final, con un learning rate más bajo.

    Al bajar el learning rate cada árbol aporta menos, así que se escala la
    cantidad de árboles en la misma proporción.
    """
    best = study.best_trial
    n_trees = int(best.user_attrs["n_estimators"] * SEARCH_LR / learning_rate)
    return {**best.params, "learning_rate": learning_rate, "n_estimators": n_trees}
