"""Parte 2 del TP: tasación de vivienda residencial con datos de Properati.

Problema de negocio
-------------------
Una inversora que compra para refaccionar y revender necesita decidir qué
propiedades publicadas comprar. El modelo estima el valor de mercado de una
vivienda a partir de sus características y ubicación; las propiedades
publicadas muy por debajo de ese valor son candidatas a oportunidad.

Supuesto explícito: un precio por debajo del estimado puede ser una
oportunidad o puede reflejar un defecto que el modelo no observa (estado,
problemas legales, orientación). El modelo prioriza visitas, no decide.

Uso:  python -m src.parte2_properati
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- Configuración ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT.parent / "entrenamiento.csv"          # datos crudos, fuera del repo
OUT = ROOT / "resultados_parte2"                 # tablas y figuras
CORTE = "2020-04-01"                             # split temporal (train < corte)
PRECIO_MIN, PRECIO_MAX = 20_000, 1_000_000       # alcance: vivienda residencial
TOP_BARRIOS = 200                                # resto va a la categoría "Otro"
SEED = 42

FEATURES = ["superficie", "rooms", "bedrooms", "bathrooms", "lat", "lon",
            "property_type", "l2", "l3"]
CATEGORICAS = ["property_type", "l2", "l3"]


def cargar_y_filtrar() -> pd.DataFrame:
    """Lee el CSV y aplica los recortes que definen el alcance del modelo."""
    df = pd.read_csv(CSV)
    filas = {"crudo": len(df)}

    df = df[(df["l1"] == "Argentina") & (df["operation_type"] == "Venta")]
    filas["Argentina + venta"] = len(df)

    df = df[df["currency"] == "USD"]              # evita mezclar monedas y TC
    filas["solo USD"] = len(df)

    df = df[df["property_type"].isin(["Casa", "Departamento", "PH"])]
    filas["vivienda residencial"] = len(df)

    df = df[df["price"].between(PRECIO_MIN, PRECIO_MAX)]
    filas[f"precio {PRECIO_MIN:,}-{PRECIO_MAX:,}"] = len(df)

    # Una sola superficie: la total y, si falta, la cubierta.
    df = df.copy()
    df["superficie"] = df["surface_total"].fillna(df["surface_covered"])
    df = df[df["superficie"].notna()]
    filas["con superficie"] = len(df)

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

    pd.Series(filas, name="filas").to_csv(OUT / "01_filtros.csv")
    print(pd.Series(filas, name="filas"), "\n")
    return df


def partir_temporal(df: pd.DataFrame):
    """Entrena con los avisos viejos y evalúa con los nuevos (como en producción)."""
    train = df[df["start_date"] < CORTE]
    test = df[df["start_date"] >= CORTE]

    X_train, X_test = train[FEATURES].copy(), test[FEATURES].copy()

    # El top de barrios se calcula SOLO con train: usar test sería leakage.
    top = X_train["l3"].value_counts().nlargest(TOP_BARRIOS).index
    for X in (X_train, X_test):
        X["l3"] = X["l3"].astype(str).where(X["l3"].isin(top), "Otro")
        for col in CATEGORICAS:
            X[col] = X[col].astype("category")

    return X_train, train["price"], X_test, test["price"], test


def evaluar(y_true, y_pred, nombre) -> dict:
    """MAE en dólares (interpretable) y MAPE en porcentaje (comparable entre rangos)."""
    return {"modelo": nombre,
            "MAE_usd": mean_absolute_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = cargar_y_filtrar()
    X_train, y_train, X_test, y_test, test = partir_temporal(df)
    print(f"train: {len(X_train):,} | test: {len(X_test):,}\n")

    # 1. Baseline: predecir siempre la mediana de entrenamiento.
    pred_base = np.full(len(y_test), y_train.median())

    # 2. Prototipo out-of-the-box: boosting de árboles, sin tunear.
    modelo = HistGradientBoostingRegressor(categorical_features="from_dtype",
                                           random_state=SEED)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    # 2b. Misma receta pero prediciendo el logaritmo del precio.
    # El precio es muy asimétrico (pocas propiedades carísimas). En escala
    # logarítmica la distribución se vuelve casi simétrica y el modelo deja de
    # arrastrar las predicciones hacia el centro. Se entrena con log(precio) y
    # se vuelve a dólares con la exponencial para poder comparar las métricas.
    modelo_log = HistGradientBoostingRegressor(categorical_features="from_dtype",
                                               random_state=SEED)
    modelo_log.fit(X_train, np.log(y_train))
    pred_log = np.exp(modelo_log.predict(X_test))

    resultados = pd.DataFrame([evaluar(y_test, pred_base, "Baseline (mediana)"),
                               evaluar(y_test, pred, "HistGradientBoosting"),
                               evaluar(y_test, pred_log, "HistGradientBoosting (log precio)")])
    resultados["mejora_MAE_%"] = (1 - resultados["MAE_usd"] /
                                  resultados["MAE_usd"].iloc[0]) * 100
    resultados.to_csv(OUT / "02_resultados.csv", index=False)
    print(resultados.to_string(index=False), "\n")

    # 3. ¿Dónde falla? Error por rango de precio.
    por_rango = (pd.DataFrame({"real": y_test.values, "pred": pred})
                 .assign(pred_log=pred_log,
                         rango=pd.cut(y_test.values,
                                      [20_000, 80_000, 150_000, 300_000, 1_000_000]))
                 .groupby("rango", observed=True)
                 .apply(lambda g: pd.Series({
                     "n": len(g),
                     "MAE_usd": mean_absolute_error(g["real"], g["pred"]),
                     "MAPE": mean_absolute_percentage_error(g["real"], g["pred"]),
                     "MAPE_log": mean_absolute_percentage_error(g["real"], g["pred_log"]),
                     "sesgo_usd": (g["pred"] - g["real"]).mean(),
                     "sesgo_usd_log": (g["pred_log"] - g["real"]).mean()}),
                     include_groups=False))
    por_rango.to_csv(OUT / "03_error_por_rango.csv")
    print(por_rango.to_string(), "\n")

    # A partir de acá se usa el modelo en escala logarítmica.
    pred = pred_log
    modelo = modelo_log

    # 4. Data shift: ¿se degrada el modelo mes a mes durante la pandemia?
    por_mes = (pd.DataFrame({"mes": test["start_date"].dt.to_period("M").astype(str),
                             "real": y_test.values, "pred": pred})
               .groupby("mes")
               .apply(lambda g: pd.Series({
                   "n": len(g),
                   "MAE_usd": mean_absolute_error(g["real"], g["pred"]),
                   "MAPE": mean_absolute_percentage_error(g["real"], g["pred"])}),
                   include_groups=False))
    por_mes.to_csv(OUT / "04_error_por_mes.csv")
    print(por_mes.to_string(), "\n")

    # 5. Qué variables usa el modelo (permutación sobre una muestra, por costo).
    muestra = X_test.sample(min(20_000, len(X_test)), random_state=SEED)
    # El modelo elegido predice log(precio), así que la importancia se mide
    # contra el log del precio real (si no, se comparan escalas distintas).
    imp = permutation_importance(modelo, muestra, np.log(y_test.loc[muestra.index]),
                                 n_repeats=3, random_state=SEED,
                                 scoring="neg_mean_absolute_error")
    importancias = (pd.Series(imp.importances_mean, index=FEATURES)
                    .sort_values(ascending=False).rename("aumento_MAE_log_al_permutar"))
    importancias.to_csv(OUT / "05_importancias.csv")
    print(importancias.to_string(), "\n")

    # 6. Candidatas a oportunidad: publicadas muy por debajo del valor estimado.
    oportunidades = (test.assign(valor_estimado=pred,
                                 brecha_pct=(pred - y_test.values) / pred)
                     .query("brecha_pct > 0.30")
                     [["l2", "l3", "property_type", "superficie", "rooms",
                       "price", "valor_estimado", "brecha_pct"]]
                     .sort_values("brecha_pct", ascending=False))
    oportunidades.head(50).to_csv(OUT / "06_oportunidades_top50.csv", index=False)
    print(f"Candidatas con brecha > 30%: {len(oportunidades):,} "
          f"({len(oportunidades) / len(test):.1%} del test)")


if __name__ == "__main__":
    main()
