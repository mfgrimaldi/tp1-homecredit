"""Feature engineering: variables nuevas a partir de application y de las tablas auxiliares.

Hay cuatro bloques, que se pueden sumar de a uno para medir cuánto aporta cada uno:
  1. application_features: ratios y transformaciones sobre la tabla principal
  2. bureau_features: créditos del cliente en otras entidades (bureau.csv)
  3. previous_features: pedidos anteriores en Home Credit (previous_application.csv)
  4. installments_features: cómo pagó las cuotas de esos créditos (installments_payments.csv)

Los bloques 2 a 4 tienen varias filas por cliente, así que se resumen (agregan) a una
fila por SK_ID_CURR y se unen a application con un left join.

Nada de esto usa TARGET, así que se puede calcular sobre train y test juntos sin leakage.
"""
import numpy as np
import pandas as pd

from src.config import ID_COL, RAW_DIR

# Valor que Home Credit usa en DAYS_EMPLOYED para "no trabaja / jubilado" (~1000 años)
DAYS_EMPLOYED_ANOM = 365243


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """División que devuelve NaN (en vez de inf) cuando el denominador es 0."""
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{name}.csv", **kwargs)


# ---------------------------------------------------------------------------
# 1. Tabla principal
# ---------------------------------------------------------------------------
def application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ratios y transformaciones a application_train/test (devuelve una copia)."""
    df = df.copy()

    # El valor anómalo de DAYS_EMPLOYED se marca en una columna aparte y se pasa a NaN,
    # para que no distorsione la escala (el imputer lo completa con la mediana)
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOM).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOM, np.nan)

    # Carga financiera: cuánto pesa el préstamo y la cuota sobre el ingreso
    df["CREDIT_INCOME_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    # Cuota / préstamo ~ 1 / plazo: una cuota alta en relación al monto es un plazo corto
    df["CREDIT_TERM"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    # Si el préstamo es mayor que el precio del bien, está financiando extras (seguros, etc.)
    df["GOODS_CREDIT_RATIO"] = _safe_div(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    # Edad y antigüedad laboral (los DAYS_* vienen en días negativos antes de la solicitud)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYED_AGE_RATIO"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

    # Scores externos: son las variables más predictivas, pero EXT_SOURCE_1 falta en ~56%
    # de los casos. Los resúmenes usan los que estén disponibles para cada cliente.
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_MAX"] = ext.max(axis=1)
    df["EXT_SOURCE_STD"] = ext.std(axis=1)
    df["EXT_SOURCE_PROD"] = ext.prod(axis=1, min_count=3)  # solo si están los 3

    # Cantidad de documentos presentados
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCS_COUNT"] = df[doc_cols].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# 2. Bureau: créditos en otras entidades
# ---------------------------------------------------------------------------
def bureau_features() -> pd.DataFrame:
    """Una fila por cliente con el resumen de sus créditos en otras entidades."""
    b = _read("bureau")
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["HAS_OVERDUE"] = (b["AMT_CREDIT_SUM_OVERDUE"] > 0).astype(int)

    agg = b.groupby(ID_COL).agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_OVERDUE_COUNT=("HAS_OVERDUE", "sum"),
        BUREAU_MAX_OVERDUE=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),   # crédito más reciente
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
    )
    # Proporción del total prestado que todavía se debe
    agg["BUREAU_DEBT_CREDIT_RATIO"] = _safe_div(agg["BUREAU_DEBT_SUM"], agg["BUREAU_CREDIT_SUM"])
    agg["BUREAU_ACTIVE_RATIO"] = _safe_div(agg["BUREAU_ACTIVE_COUNT"], agg["BUREAU_COUNT"])
    return agg.reset_index()


# ---------------------------------------------------------------------------
# 3. Pedidos anteriores en Home Credit
# ---------------------------------------------------------------------------
def previous_features() -> pd.DataFrame:
    """Una fila por cliente con el resumen de sus solicitudes anteriores."""
    p = _read("previous_application")
    # Mismo valor anómalo que en DAYS_EMPLOYED
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
              "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        p[c] = p[c].replace(DAYS_EMPLOYED_ANOM, np.nan)

    p["IS_REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["IS_APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    # Lo que pidió vs. lo que le dieron: < 1 si le otorgaron menos de lo que pidió
    p["APP_CREDIT_RATIO"] = _safe_div(p["AMT_APPLICATION"], p["AMT_CREDIT"])

    agg = p.groupby(ID_COL).agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_REFUSED_RATIO=("IS_REFUSED", "mean"),
        PREV_APPROVED_RATIO=("IS_APPROVED", "mean"),
        PREV_APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),  # pedido más reciente
    )
    return agg.reset_index()


# ---------------------------------------------------------------------------
# 4. Historial de pago de cuotas
# ---------------------------------------------------------------------------
def installments_features() -> pd.DataFrame:
    """Una fila por cliente con su comportamiento de pago de cuotas."""
    i = _read("installments_payments")
    # Días de atraso (> 0 si pagó después del vencimiento) y de anticipo
    i["DAYS_LATE"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    i["IS_LATE"] = (i["DAYS_LATE"] > 0).astype(int)
    # Pagó menos de lo que correspondía a esa cuota
    i["IS_UNDERPAID"] = (i["AMT_PAYMENT"] < i["AMT_INSTALMENT"]).astype(int)
    i["PAYMENT_RATIO"] = _safe_div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])

    agg = i.groupby(ID_COL).agg(
        INST_COUNT=("NUM_INSTALMENT_NUMBER", "count"),
        INST_LATE_RATIO=("IS_LATE", "mean"),
        INST_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INST_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INST_UNDERPAID_RATIO=("IS_UNDERPAID", "mean"),
        INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
    )
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Armado final
# ---------------------------------------------------------------------------
BLOCKS = {
    "bureau": bureau_features,
    "previous": previous_features,
    "installments": installments_features,
}


def build_features(train: pd.DataFrame, test: pd.DataFrame,
                   blocks: tuple = ("bureau", "previous", "installments")
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica application_features y une los bloques pedidos a train y test.

    Los clientes sin registros en una tabla quedan con NaN en esas columnas
    (el imputer del Pipeline los completa; LightGBM los maneja directamente).
    """
    train, test = application_features(train), application_features(test)
    for name in blocks:
        agg = BLOCKS[name]()
        train = train.merge(agg, on=ID_COL, how="left")
        test = test.merge(agg, on=ID_COL, how="left")
    return train, test
