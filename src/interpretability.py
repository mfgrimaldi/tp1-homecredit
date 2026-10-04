"""Interpretabilidad del modelo LightGBM: importancia de variables y valores SHAP.

Dos miradas complementarias:
  * Importancia por ganancia (gain): cuánto mejoró el modelo, sumado sobre todos
    los cortes que usan cada variable. Dice QUÉ variables usa más, pero no en qué
    dirección empujan.
  * SHAP: para cada cliente, cuánto sumó o restó cada variable a su predicción
    (en escala log-odds) respecto del promedio. Dice QUÉ pesa y HACIA DÓNDE:
    p. ej. "un EXT_SOURCE_MEAN bajo sube el riesgo de este cliente".

LightGBM calcula los SHAP exactos de sus árboles con predict(pred_contrib=True),
así que no hace falta aproximar.
"""
import pandas as pd
from sklearn.pipeline import Pipeline


def fit_full(build_model, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Entrena el modelo con todo train (para interpretar, no para evaluar)."""
    model = build_model()
    model.fit(X, y)
    return model


def gain_importance(model: Pipeline) -> pd.DataFrame:
    """Importancia por ganancia de cada variable, en % del total."""
    booster = model.named_steps["clf"].booster_
    imp = pd.DataFrame({
        "variable": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
        "splits": booster.feature_importance(importance_type="split"),
    })
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)


def shap_values(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Valores SHAP por cliente y variable (sin la columna del valor base)."""
    Xt = model.named_steps["prep"].transform(X)
    contrib = model.named_steps["clf"].booster_.predict(Xt, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=Xt.columns, index=X.index)


def shap_importance(shap_df: pd.DataFrame) -> pd.Series:
    """Importancia global según SHAP: promedio del valor absoluto por variable."""
    return shap_df.abs().mean().sort_values(ascending=False)
