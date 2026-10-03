"""Features del bloque bureau (mi parte del reparto del grupo: prefijo BUR_).

Qué es cada tabla
-----------------
* bureau.csv (1,7M filas): un credito que el cliente tuvo en OTRA entidad,
  reportado al Credit Bureau. Varias filas por SK_ID_CURR.
* bureau_balance.csv (27,3M filas): el estado MES a MES de cada uno de esos
  creditos. Varias filas por SK_ID_BUREAU, que a su vez tiene varias por cliente.

Por eso la agregacion es en DOS niveles: primero resumo el historial mensual por
credito (SK_ID_BUREAU) y recien despues agrego todo por cliente (SK_ID_CURR).
Agregar siempre antes de unir: si hiciera el merge directo, cada prestamo de
application quedaria duplicado tantas veces como meses de historial tiene.

Decisiones
----------
* bureau_balance se lee por chunks (son 375 MB) y se resume por credito en cada
  chunk. Los resumenes de cada chunk se combinan despues con sumas, minimos y
  maximos, que son operaciones asociativas: da igual en que chunk cayo cada fila.
* STATUS es categorico: C = cerrado, X = sin informacion, 0 a 5 = tramos de dias
  de atraso. Lo convierto a un numero (dpd) para poder promediar y tomar maximos,
  dejando C y X como NaN para no inventar un atraso de cero donde no hay dato.
* No hay leakage: ninguna de estas features mira el TARGET, solo el historial
  crediticio del cliente. Lo que si tiene que ir dentro del pipeline (imputar,
  escalar) se hace despues, en el modelo.

Salida: data/features_bureau.parquet, UNA fila por SK_ID_CURR, todas las columnas
con prefijo BUR_. Se une al resto con un merge por SK_ID_CURR.

Uso (desde la raiz del repo):
    python -m src.features_bureau
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, ID_COL, RAW_DIR

SALIDA = DATA_DIR / "features_bureau.parquet"
PREFIJO = "BUR_"
CHUNK = 3_000_000          # filas de bureau_balance por chunk (RAM acotada)
CREDITO_COL = "SK_ID_BUREAU"

# STATUS -> dias de atraso aproximados. C (cerrado) y X (sin info) quedan NaN.
DPD = {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0}

DTYPES_BALANCE = {CREDITO_COL: "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"}


# ---------------------------------------------------------------------------
# Nivel 1: bureau_balance -> una fila por credito (SK_ID_BUREAU)
# ---------------------------------------------------------------------------
def _dpd_numerico(status: pd.Series) -> np.ndarray:
    """Traduce STATUS (categorico) a dias de atraso, sin recorrer fila por fila."""
    tabla = np.array([DPD.get(c, np.nan) for c in status.cat.categories], dtype="float32")
    codigos = status.cat.codes.to_numpy()
    # cat.codes usa -1 para NaN: lo mando a NaN en vez de leer el ultimo elemento
    return np.where(codigos >= 0, tabla[np.clip(codigos, 0, None)], np.nan)


def _resumir_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Resume un chunk de bureau_balance por credito (resultados combinables)."""
    chunk = chunk.copy()
    chunk["dpd"] = _dpd_numerico(chunk["STATUS"])
    chunk["mora"] = chunk["dpd"] > 0
    chunk["mora_12m"] = chunk["mora"] & (chunk["MONTHS_BALANCE"] >= -12)
    chunk["cerrado"] = chunk["STATUS"].astype(str).eq("C")
    chunk["sin_info"] = chunk["STATUS"].astype(str).eq("X")

    g = chunk.groupby(CREDITO_COL, observed=True)
    return pd.DataFrame({
        "meses_n": g.size(),
        "mes_min": g["MONTHS_BALANCE"].min(),      # mes mas viejo (negativo)
        "mes_max": g["MONTHS_BALANCE"].max(),      # mes mas reciente
        "dpd_max": g["dpd"].max(),
        "dpd_suma": g["dpd"].sum(),
        "meses_mora": g["mora"].sum(),
        "meses_mora_12m": g["mora_12m"].sum(),
        "meses_cerrado": g["cerrado"].sum(),
        "meses_sin_info": g["sin_info"].sum(),
    })


def resumir_balance(path: Path | None = None, chunksize: int = CHUNK) -> pd.DataFrame:
    """Lee bureau_balance por partes y devuelve una fila por SK_ID_BUREAU."""
    path = path or RAW_DIR / "bureau_balance.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")

    partes = []
    for i, chunk in enumerate(pd.read_csv(path, dtype=DTYPES_BALANCE, chunksize=chunksize), 1):
        partes.append(_resumir_chunk(chunk))
        print(f"  chunk {i}: {len(chunk):,} filas -> {len(partes[-1]):,} creditos")

    # Un mismo credito puede aparecer en dos chunks: se combinan las parciales
    bal = pd.concat(partes).groupby(level=0).agg(
        meses_n=("meses_n", "sum"),
        mes_min=("mes_min", "min"),
        mes_max=("mes_max", "max"),
        dpd_max=("dpd_max", "max"),
        dpd_suma=("dpd_suma", "sum"),
        meses_mora=("meses_mora", "sum"),
        meses_mora_12m=("meses_mora_12m", "sum"),
        meses_cerrado=("meses_cerrado", "sum"),
        meses_sin_info=("meses_sin_info", "sum"),
    )

    # Normalizo por cantidad de meses: una mora de 3 meses sobre 6 no es lo mismo
    # que sobre 60. Las proporciones son comparables entre creditos.
    bal["tasa_mora"] = bal["meses_mora"] / bal["meses_n"]
    bal["frac_cerrado"] = bal["meses_cerrado"] / bal["meses_n"]
    bal["frac_sin_info"] = bal["meses_sin_info"] / bal["meses_n"]
    bal["dpd_prom"] = bal["dpd_suma"] / bal["meses_n"]
    bal["antiguedad_meses"] = -bal["mes_min"]       # cuantos meses de historial
    bal["meses_sin_reportar"] = -bal["mes_max"]     # hace cuanto no se reporta
    return bal.drop(columns=["dpd_suma", "mes_min", "mes_max", "meses_cerrado", "meses_sin_info"])


# ---------------------------------------------------------------------------
# Nivel 2: bureau (+ resumen del balance) -> una fila por cliente (SK_ID_CURR)
# ---------------------------------------------------------------------------
# Columnas del balance que ya vienen resumidas por credito y se vuelven a agregar
COLS_BALANCE = {
    "meses_n": ["sum", "mean", "max"],
    "tasa_mora": ["mean", "max"],
    "dpd_max": ["mean", "max"],
    "dpd_prom": ["mean"],
    "meses_mora": ["sum", "max"],
    "meses_mora_12m": ["sum", "max"],
    "antiguedad_meses": ["max", "mean"],
    "meses_sin_reportar": ["min", "mean"],
    "frac_cerrado": ["mean"],
    "frac_sin_info": ["mean"],
}

COLS_BUREAU = {
    "DAYS_CREDIT": ["min", "max", "mean", "std"],        # antiguedad de los creditos
    "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],       # vencimientos
    "DAYS_ENDDATE_FACT": ["max", "mean"],                # cierres efectivos
    "DAYS_CREDIT_UPDATE": ["max", "mean"],               # frescura de la info
    "CREDIT_DAY_OVERDUE": ["max", "mean"],               # atraso vigente
    "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],           # peor atraso historico
    "CNT_CREDIT_PROLONG": ["sum", "max"],                # refinanciaciones
    "AMT_CREDIT_SUM": ["sum", "mean", "max"],            # monto otorgado
    "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],       # deuda viva
    "AMT_CREDIT_SUM_LIMIT": ["sum", "mean"],             # limite disponible
    "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],            # monto en mora
    "AMT_ANNUITY": ["sum", "mean", "max"],               # cuotas
}


def _aplanar(cols) -> list:
    return [f"{c}_{a}".upper() for c, a in cols]


def _agregar(bureau: pd.DataFrame, columnas: dict, sufijo: str = "") -> pd.DataFrame:
    """Agrega por cliente las columnas numericas indicadas."""
    presentes = {c: a for c, a in columnas.items() if c in bureau.columns}
    out = bureau.groupby(ID_COL).agg(presentes)
    out.columns = [f"{n}{sufijo}" for n in _aplanar(out.columns)]
    return out


def construir_features(bureau: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """Une el resumen por credito con bureau y agrega todo por cliente."""
    b = bureau.merge(balance, how="left", left_on=CREDITO_COL, right_index=True)
    if len(b) != len(bureau):
        raise ValueError("El merge con el balance cambio la cantidad de filas de bureau")

    # Banderas por credito, para poder contarlas por cliente
    b["es_activo"] = b["CREDIT_ACTIVE"].eq("Active")
    b["es_cerrado"] = b["CREDIT_ACTIVE"].eq("Closed")
    b["es_hipoteca"] = b["CREDIT_TYPE"].eq("Mortgage")
    b["es_micro"] = b["CREDIT_TYPE"].eq("Microloan")
    b["es_tarjeta"] = b["CREDIT_TYPE"].eq("Credit card")
    b["moneda_distinta"] = ~b["CREDIT_CURRENCY"].eq("currency 1")
    b["tuvo_mora"] = b["AMT_CREDIT_MAX_OVERDUE"].fillna(0) > 0
    b["tiene_balance"] = b["meses_n"].notna()

    g = b.groupby(ID_COL)
    conteos = pd.DataFrame({
        "N_CREDITOS": g.size(),
        "N_ACTIVOS": g["es_activo"].sum(),
        "N_CERRADOS": g["es_cerrado"].sum(),
        "N_TIPOS_CREDITO": g["CREDIT_TYPE"].nunique(),
        "N_HIPOTECAS": g["es_hipoteca"].sum(),
        "N_MICROCREDITOS": g["es_micro"].sum(),
        "N_TARJETAS": g["es_tarjeta"].sum(),
        "N_MONEDA_DISTINTA": g["moneda_distinta"].sum(),
        "N_CON_MORA_HIST": g["tuvo_mora"].sum(),
        "N_CON_BALANCE": g["tiene_balance"].sum(),
    })

    feats = pd.concat([
        conteos,
        _agregar(b, COLS_BUREAU),
        _agregar(b, COLS_BALANCE),
        # Mismas agregaciones pero solo con los creditos VIGENTES: la foto de hoy
        # pesa distinto que el historial completo.
        _agregar(b[b["es_activo"]], {
            "AMT_CREDIT_SUM": ["sum", "max"],
            "AMT_CREDIT_SUM_DEBT": ["sum", "max"],
            "AMT_ANNUITY": ["sum"],
            "DAYS_CREDIT": ["max"],
            "CREDIT_DAY_OVERDUE": ["max"],
            "tasa_mora": ["mean", "max"],
            "dpd_max": ["max"],
        }, sufijo="_ACT"),
    ], axis=1)

    # --- Ratios con sentido economico ---------------------------------------
    credito_total = feats["AMT_CREDIT_SUM_SUM"].replace(0, np.nan)
    feats["RATIO_DEUDA_CREDITO"] = feats["AMT_CREDIT_SUM_DEBT_SUM"] / credito_total
    feats["RATIO_MORA_CREDITO"] = feats["AMT_CREDIT_SUM_OVERDUE_SUM"] / credito_total
    feats["RATIO_LIMITE_CREDITO"] = feats["AMT_CREDIT_SUM_LIMIT_SUM"] / credito_total
    feats["RATIO_DEUDA_ACTIVA"] = (feats["AMT_CREDIT_SUM_DEBT_SUM_ACT"]
                                   / feats["AMT_CREDIT_SUM_SUM_ACT"].replace(0, np.nan))
    feats["FRAC_ACTIVOS"] = feats["N_ACTIVOS"] / feats["N_CREDITOS"]
    feats["FRAC_CON_MORA_HIST"] = feats["N_CON_MORA_HIST"] / feats["N_CREDITOS"]
    feats["DIAS_DESDE_ULTIMO_CREDITO"] = -feats["DAYS_CREDIT_MAX"]
    feats["DIAS_DESDE_PRIMER_CREDITO"] = -feats["DAYS_CREDIT_MIN"]
    # Intensidad de endeudamiento: cuantos creditos por anio de historial
    anios = (feats["DIAS_DESDE_PRIMER_CREDITO"] / 365).replace(0, np.nan)
    feats["CREDITOS_POR_ANIO"] = feats["N_CREDITOS"] / anios
    feats["TIENE_BUREAU"] = 1     # tras el merge con application, NaN = sin historial

    feats.columns = [PREFIJO + c for c in feats.columns]
    return feats.sort_index()


def guardar(feats: pd.DataFrame, salida: Path = SALIDA) -> Path:
    """Guarda en parquet; si no hay motor de parquet, cae a csv comprimido."""
    salida.parent.mkdir(parents=True, exist_ok=True)
    try:
        feats.to_parquet(salida)
        return salida
    except ImportError:
        alternativa = salida.with_suffix(".csv.gz")
        feats.to_csv(alternativa)
        print(f"Sin pyarrow/fastparquet: guarde {alternativa.name} en vez de parquet")
        return alternativa


def main() -> None:
    print("1) Resumiendo bureau_balance por credito...")
    balance = resumir_balance()
    print(f"   {len(balance):,} creditos con historial mensual")

    print("2) Leyendo bureau...")
    bureau = pd.read_csv(RAW_DIR / "bureau.csv")
    print(f"   {len(bureau):,} creditos de {bureau[ID_COL].nunique():,} clientes")

    print("3) Agregando por cliente...")
    feats = construir_features(bureau, balance)

    # Chequeos: una fila por cliente y ninguna columna constante o vacia
    assert feats.index.is_unique, "Hay clientes repetidos en las features"
    assert feats.index.name == ID_COL
    vacias = feats.columns[feats.notna().sum() == 0].tolist()
    if vacias:
        print(f"   OJO, columnas todas NaN: {vacias}")

    ruta = guardar(feats)
    print(f"\nListo: {ruta} -> {feats.shape[0]:,} clientes x {feats.shape[1]} features")
    print(f"Faltantes promedio: {feats.isna().mean().mean():.1%}")


if __name__ == "__main__":
    main()
