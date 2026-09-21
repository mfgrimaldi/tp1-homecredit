"""Definición de modelos. Cada función devuelve un estimador SIN entrenar.

El preprocesamiento va dentro del Pipeline para que se ajuste solo con los datos
de entrenamiento de cada fold (evita leakage de medianas, medias, categorías).
"""
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic(C: float = 1.0) -> Pipeline:
    """Baseline: logística con preprocesamiento mínimo.

    * Numéricas: imputación por mediana + estandarización (la logística lo necesita
      para que la regularización trate a todas las variables por igual).
    * Categóricas: imputación con la categoría "NA" + one-hot. Las categorías poco
      frecuentes (< 1% de las filas) se agrupan para no crear columnas ruidosas.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_include=["object", "string", "category"])),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


def get_features(df: pd.DataFrame, drop: tuple = ("SK_ID_CURR", "TARGET")) -> pd.DataFrame:
    """Columnas que entran al modelo: todo menos el ID y el target."""
    return df.drop(columns=[c for c in drop if c in df.columns])
