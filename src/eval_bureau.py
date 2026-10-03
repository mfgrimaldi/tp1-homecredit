"""Cuanto suma el bloque BUR_ (bureau + bureau_balance) sobre el baseline.

Compara con los MISMOS folds del repo (data/folds.csv) dos conjuntos de features:
  * app : solo application (lo que ya teniamos)
  * bur : application + las 78 features de bureau
La diferencia de AUC es la evidencia que pide la consigna: que aporto cada tabla.

El merge es LEFT desde application: los clientes sin historial en el bureau quedan
con NaN en todas las columnas BUR_ (14% de la base), y XGBoost maneja los NaN solo.
Ninguna feature de bureau mira el TARGET, asi que calcularlas sobre toda la base no
es leakage; lo que si se ajusta dentro de cada fold es el modelo.

Cada fold se guarda en resultados_bureau/parciales/ para poder cortar la corrida
en varias tandas (son 10 modelos) y resumir al final sin repetir nada.

Uso (desde la raiz del repo):
    python -m src.eval_bureau                          # todo de una
    python -m src.eval_bureau --bloques bur --folds 0,1  # una tanda
    python -m src.eval_bureau --resumir                # tabla final + submission
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import DATA_DIR, ID_COL, TARGET_COL
from src.data import attach_folds, load_application
from src.features_app import preparar
from src.models import build_xgboost
from src.submission import make_submission

FEATURES_BUREAU = DATA_DIR / "features_bureau.parquet"
RESULTADOS = Path(__file__).resolve().parents[1] / "resultados_bureau"
PARCIALES = RESULTADOS / "parciales"
BLOQUES = ("app", "bur")
NOMBRES = {"app": "application", "bur": "application + BUR_"}


def _aligerar(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32: la mitad de memoria, misma informacion para el modelo."""
    flotantes = df.select_dtypes("float64").columns
    df[flotantes] = df[flotantes].astype("float32")
    return df


def preparar_datos() -> dict:
    """Devuelve las matrices de los dos bloques, el target, los folds y los IDs."""
    train, test = load_application("train"), load_application("test")
    y, folds = train[TARGET_COL], attach_folds(train)
    ids_train, ids_test = train[ID_COL], test[ID_COL]

    X, X_test = preparar(train, test)
    X, X_test = _aligerar(X), _aligerar(X_test)

    if not FEATURES_BUREAU.exists():
        raise FileNotFoundError(f"Falta {FEATURES_BUREAU}. Corre: python -m src.features_bureau")
    bur = _aligerar(pd.read_parquet(FEATURES_BUREAU))

    # reindex por SK_ID_CURR y despues alineo con el indice de application, para
    # que el orden de las filas sea exactamente el mismo (sin merge que duplique)
    Xb = X.join(bur.reindex(ids_train.to_numpy()).set_index(X.index))
    Xb_test = X_test.join(bur.reindex(ids_test.to_numpy()).set_index(X_test.index))
    if len(Xb) != len(X) or len(Xb_test) != len(X_test):
        raise ValueError("La union con bureau cambio la cantidad de filas")

    print(f"application: {X.shape[1]} columnas | + bureau: {Xb.shape[1]} columnas")
    print(f"clientes con historial de bureau: {Xb['BUR_TIENE_BUREAU'].notna().mean():.1%}\n")
    return {"app": (X, X_test), "bur": (Xb, Xb_test),
            "y": y, "folds": folds, "ids_test": ids_test}


def correr_fold(datos: dict, bloque: str, k: int, trees: int) -> None:
    """Entrena el fold k de un bloque y guarda sus predicciones."""
    X, X_test = datos[bloque]
    y, folds = datos["y"], datos["folds"]
    tr, va = (folds != k).to_numpy(), (folds == k).to_numpy()

    t0 = time.time()
    modelo = build_xgboost(n_estimators=trees).fit(X[tr], y[tr])
    pred_va = modelo.predict_proba(X[va])[:, 1]
    pred_test = modelo.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y[va], pred_va)

    PARCIALES.mkdir(parents=True, exist_ok=True)
    np.savez(PARCIALES / f"{bloque}_fold{k}.npz", va=np.flatnonzero(va),
             pred_va=pred_va, pred_test=pred_test, auc=auc)
    print(f"{bloque} fold {k}: AUC = {auc:.4f} ({time.time() - t0:.0f}s)")


def resumir(datos: dict, trees: int, submission: bool = True) -> pd.DataFrame:
    """Junta los folds guardados, calcula el AUC OOF y arma la tabla del informe."""
    y = datos["y"]
    filas, test_preds = [], {}

    for bloque in BLOQUES:
        archivos = sorted(PARCIALES.glob(f"{bloque}_fold*.npz"))
        if not archivos:
            continue
        oof = np.full(len(y), np.nan)
        test = np.zeros(len(datos[bloque][1]))
        aucs = []
        for a in archivos:
            d = np.load(a)
            oof[d["va"]] = d["pred_va"]
            test += d["pred_test"] / len(archivos)
            aucs.append(float(d["auc"]))
        hay = ~np.isnan(oof)
        filas.append({"bloque": NOMBRES[bloque],
                      "n_features": datos[bloque][0].shape[1],
                      "folds": len(archivos),
                      "auc_oof": roc_auc_score(y[hay], oof[hay]),
                      "auc_medio_fold": np.mean(aucs),
                      "desvio_fold": np.std(aucs)})
        test_preds[bloque] = test

    tabla = pd.DataFrame(filas)
    tabla["aporte_auc"] = tabla["auc_oof"] - tabla["auc_oof"].iloc[0]
    RESULTADOS.mkdir(exist_ok=True)
    tabla.to_csv(RESULTADOS / "aporte_bloque_bureau.csv", index=False)
    print("\n" + tabla.to_string(index=False))

    # Que features usa el modelo: importancia por ganancia sobre toda la base.
    # (La de permutacion, que es la que pide la consigna, va en el notebook de
    # interpretabilidad: es la misma idea pero mucho mas caro de correr.)
    X, _ = datos["bur"]
    final = build_xgboost(n_estimators=trees).fit(X, y)
    imp = (pd.Series(final.get_booster().get_score(importance_type="gain"))
           .sort_values(ascending=False).rename("ganancia"))
    imp.to_csv(RESULTADOS / "importancia_ganancia.csv")
    print("\nTop 15 features (ganancia):")
    print(imp.head(15).to_string())
    print(f"\nFeatures BUR_ en el top 30: {sum(c.startswith('BUR_') for c in imp.head(30).index)}/30")

    if submission and "bur" in test_preds:
        auc = tabla.loc[tabla["bloque"] == NOMBRES["bur"], "auc_oof"].iloc[0]
        make_submission(datos["ids_test"], test_preds["bur"], "xgb_app_bureau", auc)
    return tabla


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloques", default="app,bur", help="app, bur o las dos")
    ap.add_argument("--folds", default="", help="folds a correr, ej: 0,1 (vacio = todos)")
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--resumir", action="store_true", help="solo juntar lo ya corrido")
    ap.add_argument("--sin-submission", action="store_true")
    args = ap.parse_args()

    datos = preparar_datos()
    if not args.resumir:
        folds = ([int(f) for f in args.folds.split(",")] if args.folds
                 else sorted(datos["folds"].unique()))
        for bloque in args.bloques.split(","):
            for k in folds:
                correr_fold(datos, bloque, k, args.trees)
    resumir(datos, args.trees, submission=not args.sin_submission)


if __name__ == "__main__":
    main()
