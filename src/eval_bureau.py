"""Cuanto aporta el bloque BUR_ (bureau + bureau_balance) al modelo del grupo.

El repo ya tiene un bloque de bureau en src/features.py (conteos, deuda, maximo
atraso, dias desde el ultimo credito) que resume solo bureau.csv. Este modulo
compara ese bloque contra las 78 features de src/features_bureau.py, que ademas
usan bureau_balance.csv, el historial MES a MES de cada credito.

Cuatro variantes, siempre con los mismos folds de data/folds.csv y el mismo
LightGBM (los hiperparametros optimizados de resultados/lgbm_best_params.json):

    sin_bureau   application + previous + installments
    bureau_viejo + el bloque bureau de src/features.py
    bureau_nuevo + el bloque BUR_ (bureau_balance incluido)
    ambos        los dos bloques de bureau juntos

Ninguna feature mira el TARGET, asi que calcularlas sobre train y test no es
leakage; lo que se ajusta dentro de cada fold es el modelo.

Las agregaciones de previous e installments tardan varios minutos (1,1 GB de
CSV), asi que se cachean en data/cache/ la primera vez.

Uso (desde la raiz del repo):
    python -m src.eval_bureau                             # las 4 variantes, 5 folds
    python -m src.eval_bureau --variantes bureau_nuevo --folds 0,1
    python -m src.eval_bureau --resumir                   # tabla final + submission
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import DATA_DIR, ID_COL, TARGET_COL
from src.data import attach_folds, load_application
from src.features import BLOCKS, application_features
from src.features_bureau import SALIDA as FEATURES_BUREAU
from src.models import build_lgbm
from src.submission import make_submission

CACHE = DATA_DIR / "cache"
RESULTADOS = Path(__file__).resolve().parents[1] / "resultados_bureau"
PARCIALES = RESULTADOS / "parciales"
PARAMS_PATH = Path(__file__).resolve().parents[1] / "resultados" / "lgbm_best_params.json"

VARIANTES = {
    "sin_bureau":   {"aggs": ("previous", "installments"), "bur": False},
    "bureau_viejo": {"aggs": ("bureau", "previous", "installments"), "bur": False},
    "bureau_nuevo": {"aggs": ("previous", "installments"), "bur": True},
    "ambos":        {"aggs": ("bureau", "previous", "installments"), "bur": True},
}


def _aligerar(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32: la mitad de memoria, misma informacion para el modelo."""
    flotantes = df.select_dtypes("float64").columns
    df[flotantes] = df[flotantes].astype("float32")
    return df


def agg_cacheada(nombre: str) -> pd.DataFrame:
    """Agregado por cliente de una tabla auxiliar, calculado una sola vez."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"agg_{nombre}.parquet"
    if not path.exists():
        t0 = time.time()
        BLOCKS[nombre]().to_parquet(path, index=False)
        print(f"  agg {nombre}: calculado en {time.time() - t0:.0f}s")
    return pd.read_parquet(path)


def armar(variante: str) -> dict:
    """Devuelve X, X_test, y, folds e ids para una variante."""
    cfg = VARIANTES[variante]
    train, test = load_application("train"), load_application("test")
    y, folds = train[TARGET_COL], attach_folds(train)
    ids_test = test[ID_COL]

    train, test = application_features(train), application_features(test)
    for nombre in cfg["aggs"]:
        agg = agg_cacheada(nombre)
        train = train.merge(agg, on=ID_COL, how="left")
        test = test.merge(agg, on=ID_COL, how="left")

    if cfg["bur"]:
        bur = pd.read_parquet(FEATURES_BUREAU).reset_index()
        train = train.merge(bur, on=ID_COL, how="left")
        test = test.merge(bur, on=ID_COL, how="left")

    quitar = [c for c in (ID_COL, TARGET_COL) if c in train.columns]
    X = _aligerar(train.drop(columns=quitar))
    X_test = _aligerar(test.drop(columns=[c for c in quitar if c in test.columns]))[X.columns]
    if len(X) != len(y):
        raise ValueError("Algun merge duplico filas de application")
    print(f"{variante}: {X.shape[1]} columnas")
    return {"X": X, "X_test": X_test, "y": y, "folds": folds, "ids_test": ids_test}


def params_lgbm(lr: float | None = None) -> dict:
    """Hiperparametros optimizados con Optuna (PR #2). Si no estan, usa los default.

    El modelo optimizado son 5892 arboles con learning_rate 0.02: para la tabla de
    ablacion, donde lo que importa es la DIFERENCIA entre variantes y no el ultimo
    decimal del AUC, se puede subir el learning rate y bajar los arboles en la misma
    proporcion (--lr 0.05 => ~2350 arboles). Es la misma receta para las cuatro
    variantes, asi que la comparacion sigue siendo justa, y tarda un tercio.
    """
    if not PARAMS_PATH.exists():
        print("OJO: no encontre lgbm_best_params.json, uso los parametros por defecto")
        return {}
    params = json.loads(PARAMS_PATH.read_text())
    if lr:
        params["n_estimators"] = int(params["n_estimators"] * params["learning_rate"] / lr)
        params["learning_rate"] = lr
        print(f"ablacion rapida: learning_rate {lr} con {params['n_estimators']} arboles")
    return params


def correr_fold(datos: dict, variante: str, k: int, params: dict, rehacer: bool = False) -> None:
    """Entrena un fold y guarda sus predicciones, para poder cortar la corrida.

    Si el fold ya esta guardado no se vuelve a entrenar (cada uno tarda minutos):
    asi se puede relanzar el comando completo y sigue donde quedo. Con --rehacer
    se ignora lo guardado y se entrena todo de nuevo.
    """
    destino = PARCIALES / f"{variante}_fold{k}.npz"
    if destino.exists() and not rehacer:
        print(f"{variante} fold {k}: ya estaba corrido (AUC = {float(np.load(destino)['auc']):.4f})")
        return
    X, y, folds = datos["X"], datos["y"], datos["folds"]
    tr, va = (folds != k).to_numpy(), (folds == k).to_numpy()

    t0 = time.time()
    modelo = build_lgbm(categorical="native", **params).fit(X[tr], y[tr])
    pred_va = modelo.predict_proba(X[va])[:, 1]
    pred_test = modelo.predict_proba(datos["X_test"])[:, 1]
    auc = roc_auc_score(y[va], pred_va)

    PARCIALES.mkdir(parents=True, exist_ok=True)
    np.savez(destino, va=np.flatnonzero(va),
             pred_va=pred_va, pred_test=pred_test, auc=auc, n_features=X.shape[1])
    print(f"{variante} fold {k}: AUC = {auc:.4f} ({time.time() - t0:.0f}s)")


def resumir(y: pd.Series, ids_test: pd.Series | None = None) -> pd.DataFrame:
    """Junta los folds guardados y arma la tabla de aporte para el informe."""
    filas, test_preds = [], {}
    for variante in VARIANTES:
        archivos = sorted(PARCIALES.glob(f"{variante}_fold*.npz"))
        if not archivos:
            continue
        oof = np.full(len(y), np.nan)
        test, aucs, n_features = None, [], None
        for a in archivos:
            d = np.load(a)
            oof[d["va"]] = d["pred_va"]
            test = d["pred_test"] / len(archivos) if test is None else test + d["pred_test"] / len(archivos)
            aucs.append(float(d["auc"]))
            n_features = int(d["n_features"])
        hay = ~np.isnan(oof)
        filas.append({"variante": variante, "n_features": n_features, "folds": len(archivos),
                      "auc_oof": roc_auc_score(y[hay], oof[hay]),
                      "auc_medio_fold": float(np.mean(aucs)),
                      "desvio_fold": float(np.std(aucs))})
        test_preds[variante] = test

    tabla = pd.DataFrame(filas)
    if "sin_bureau" in tabla["variante"].values:
        base = tabla.loc[tabla["variante"] == "sin_bureau", "auc_oof"].iloc[0]
        tabla["aporte_auc"] = tabla["auc_oof"] - base
    RESULTADOS.mkdir(exist_ok=True)
    tabla.to_csv(RESULTADOS / "aporte_bloque_bureau.csv", index=False)
    print("\n" + tabla.to_string(index=False))

    mejor = tabla.sort_values("auc_oof").iloc[-1]
    if ids_test is not None and mejor["variante"] in test_preds and mejor["folds"] == 5:
        make_submission(ids_test, test_preds[mejor["variante"]],
                        f"lgbm_{mejor['variante']}", mejor["auc_oof"])
    return tabla


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variantes", default=",".join(VARIANTES))
    ap.add_argument("--folds", default="", help="folds a correr, ej: 0,1 (vacio = todos)")
    ap.add_argument("--lr", type=float, default=0.0,
                    help="sube el learning rate y baja los arboles en proporcion (ej: 0.05)")
    ap.add_argument("--rehacer", action="store_true", help="reentrenar aunque el fold ya este guardado")
    ap.add_argument("--resumir", action="store_true", help="solo juntar lo ya corrido")
    args = ap.parse_args()

    if args.resumir:
        train = load_application("train", usecols=[ID_COL, TARGET_COL])
        resumir(train[TARGET_COL], None)
        return

    params = params_lgbm(args.lr)
    for variante in args.variantes.split(","):
        folds_pedidos = ([int(f) for f in args.folds.split(",")] if args.folds else list(range(5)))
        if not args.rehacer and all((PARCIALES / f"{variante}_fold{k}.npz").exists() for k in folds_pedidos):
            print(f"{variante}: ya estaban los {len(folds_pedidos)} folds, no se reentrena")
            continue
        datos = armar(variante)
        folds = ([int(f) for f in args.folds.split(",")] if args.folds
                 else sorted(datos["folds"].unique()))
        for k in folds:
            correr_fold(datos, variante, k, params, args.rehacer)

    train = load_application("train", usecols=[ID_COL, TARGET_COL])
    ids_test = load_application("test", usecols=[ID_COL])[ID_COL]
    resumir(train[TARGET_COL], ids_test)


if __name__ == "__main__":
    main()
