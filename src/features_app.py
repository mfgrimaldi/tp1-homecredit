"""Preprocesamiento de application_{train,test} (mi limpieza previa al FE de Alber).

Decisiones del grupo (ver README / informe):
  * Se eliminan los bloques de columnas que se consideraron sin sentido para el
    problema: auto, datos de contacto, características del edificio y consultas
    al Credit Bureau.
  * DAYS_EMPLOYED = 365243 es un placeholder (≈1000 años de antigüedad) que
    aparece exactamente en las mismas filas que ORGANIZATION_TYPE = 'XNA'. Son
    jubilados (55.352 filas) y unos pocos desempleados (22). Se lo explicitamos
    al modelo con una flag y una categoría propia, y el valor falso pasa a NaN.
"""
import pandas as pd

# ---------------------------------------------------------------------------
# Columnas a eliminar
# ---------------------------------------------------------------------------
DROP_AUTO = ["FLAG_OWN_CAR", "OWN_CAR_AGE"]

DROP_CONTACTO = [
    "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
]

# 14 medidas del edificio x 3 estadísticos (_AVG, _MODE, _MEDI) + 5 columnas sueltas
_MEDIDAS_EDIFICIO = [
    "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
    "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
    "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA",
]
DROP_EDIFICIO = [f"{m}_{s}" for s in ("AVG", "MODE", "MEDI") for m in _MEDIDAS_EDIFICIO] + [
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE",
    "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
]

DROP_CREDIT_BUREAU = [
    f"AMT_REQ_CREDIT_BUREAU_{p}" for p in ("HOUR", "DAY", "WEEK", "MON", "QRT", "YEAR")
]

DROP_COLS = DROP_AUTO + DROP_CONTACTO + DROP_EDIFICIO + DROP_CREDIT_BUREAU

# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------
DAYS_EMPLOYED_PLACEHOLDER = 365243
ORG_TYPE_PLACEHOLDER = "XNA"
CATEGORIA_JUBILADO = "Jubilado"


def tratar_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Marca a los jubilados en vez de dejar el valor falso de 1000 años."""
    df = df.copy()
    es_jubilado = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_PLACEHOLDER

    # Flag explícita: 1 = jubilado (placeholder), 0 = trabaja
    df["FLAG_JUBILADO"] = es_jubilado.astype(int)
    # La antigüedad laboral no existe para ellos: NaN (XGBoost maneja NaN nativamente)
    df.loc[es_jubilado, "DAYS_EMPLOYED"] = float("nan")
    # En la organización, 'XNA' pasa a ser una categoría con nombre propio
    df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace(ORG_TYPE_PLACEHOLDER, CATEGORIA_JUBILADO)
    return df


def preparar(train: pd.DataFrame, test: pd.DataFrame, drop_cols=DROP_COLS):
    """Aplica todo el preprocesamiento y devuelve (X_train, X_test).

    Las categóricas quedan como dtype 'category' con las MISMAS categorías en
    train y test, para que XGBoost las use de forma nativa (enable_categorical).
    """
    train, test = tratar_placeholders(train), tratar_placeholders(test)
    quitar = [c for c in list(drop_cols) + ["SK_ID_CURR", "TARGET"] if c in train.columns]
    X, X_test = train.drop(columns=quitar), test.drop(columns=[c for c in quitar if c in test.columns])
    X_test = X_test[X.columns]

    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("str", "string"):
            cats = sorted(set(X[c].dropna()) | set(X_test[c].dropna()))
            X[c] = pd.Categorical(X[c], categories=cats)
            X_test[c] = pd.Categorical(X_test[c], categories=cats)
    return X, X_test
