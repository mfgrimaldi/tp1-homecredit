"""Definición de modelos. Cada función devuelve un estimador SIN entrenar.

El preprocesamiento va dentro del Pipeline para que se ajuste solo con los datos
de entrenamiento de cada fold (evita leakage de medianas, medias, categorías).
"""
import re

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import SEED

# Columnas de texto (categóricas) en pandas 2 y 3
CAT_DTYPES = ["object", "string", "category"]

# Hiperparámetros de LightGBM antes de optimizar: valores estándar razonables
LGBM_DEFAULTS = dict(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,          # cada árbol ve el 80% de las filas...
    subsample_freq=1,
    colsample_bytree=0.5,   # ...y el 50% de las columnas: reduce sobreajuste
    reg_lambda=1.0,
)


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


def build_xgboost(**params):
    """XGBoost con categóricas nativas (requiere columnas dtype 'category').

    No necesita imputar ni escalar: maneja NaN solo y los árboles son invariantes
    a la escala. Los hiperparámetros son un punto de partida razonable, todavía
    sin optimizar (eso va en la etapa de búsqueda de hiperparámetros).
    """
    from xgboost import XGBClassifier

    defaults = dict(
        n_estimators=600,     # en el fold 0 el AUC de validación se estabiliza ~600 árboles
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,         # fracción de filas por árbol
        colsample_bytree=0.7,  # fracción de columnas por árbol
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,   # categóricas con split óptimo por particiones
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
    )
    defaults.update(params)
    return XGBClassifier(**defaults)


def get_features(df: pd.DataFrame, drop: tuple = ("SK_ID_CURR", "TARGET")) -> pd.DataFrame:
    """Columnas que entran al modelo: todo menos el ID y el target."""
    return df.drop(columns=[c for c in drop if c in df.columns])


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Convierte las columnas de texto al tipo 'category' de pandas.

    Las categorías posibles se aprenden en fit (solo con el train del fold), así
    train y validación quedan codificados igual. Una categoría que no apareció
    en train pasa a NaN. LightGBM usa este tipo para su manejo nativo de categóricas.
    """

    def fit(self, X: pd.DataFrame, y=None):
        cols = X.select_dtypes(include=CAT_DTYPES).columns
        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, cats in self.categories_.items():
            X[c] = pd.Categorical(X[c], categories=cats)
        return X


def _clean_column_names(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM no acepta comas, comillas, etc. en los nombres de columna.

    Hace falta tras el one-hot, que crea columnas como 'NAME_TYPE_SUITE_Spouse, partner'.
    """
    return X.rename(columns=lambda c: re.sub(r"[^0-9A-Za-z_]+", "_", str(c)))


def build_lgbm(categorical: str = "native", **params) -> Pipeline:
    """LightGBM con dos opciones para las variables categóricas.

    * "native": las categóricas se pasan como tipo 'category' y LightGBM busca
      directamente la mejor forma de agrupar sus valores en cada corte.
    * "onehot": una columna 0/1 por categoría, igual que en la logística.

    A diferencia de la logística, no hace falta imputar ni estandarizar: los
    árboles manejan los NaN solos y no dependen de la escala de las variables.
    """
    clf = LGBMClassifier(**{**LGBM_DEFAULTS, **params},
                         random_state=SEED, n_jobs=-1, verbose=-1)
    if categorical == "native":
        prep = CategoryCaster()
    elif categorical == "onehot":
        prep = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                   sparse_output=False),
              make_column_selector(dtype_include=CAT_DTYPES))],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        prep = Pipeline([("onehot", prep), ("names", FunctionTransformer(_clean_column_names))])
    else:
        raise ValueError("categorical tiene que ser 'native' u 'onehot'")
    return Pipeline([("prep", prep), ("clf", clf)])
