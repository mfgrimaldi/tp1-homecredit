# TP1 · Home Credit Default Risk

Machine Learning, MiM+Analytics UTDT 2026. Primera parte del TP: predecir default de préstamos (métrica ROC AUC) con la [competencia de Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data).

## Estructura

```
data/
  raw/            CSV de Kaggle (no se suben a git)
  folds.csv       asignación SK_ID_CURR -> fold (sí se sube a git)
notebooks/
  01_baseline_logistica.ipynb
src/
  config.py       rutas, semilla (42) y cantidad de folds (5)
  data.py         lectura de CSV y de folds
  make_folds.py   genera data/folds.csv (StratifiedKFold sobre TARGET)
  cv.py           validación cruzada con los folds fijos: OOF, AUC y predicción de test
  models.py       modelos (por ahora, la logística baseline)
  submission.py   arma el CSV para subir a Kaggle
  features_bureau.py  features de bureau + bureau_balance (bloque BUR_)
  eval_bureau.py      mide cuanto suma el bloque BUR_ sobre el baseline
submissions/      CSV generados (no se suben a git)
```

## Cómo arrancar

1. Instalar dependencias en tu entorno de Python: `pip install -r requirements.txt`
2. Bajar los datos de Kaggle y dejarlos en `data/raw/` (al menos `application_train.csv` y `application_test.csv`). También se pueden dejar en una carpeta `Archivos_TP/` al lado del repo, en la carpeta que contiene al repo, o en cualquier ruta indicada con la variable de entorno `TP_DATA_DIR`. `src/config.py` usa la primera que encuentre con `application_train.csv`.
3. Los folds ya están generados en `data/folds.csv`. Solo si hiciera falta regenerarlos (desde la raíz del repo): `python -m src.make_folds`
4. Correr `notebooks/01_baseline_logistica.ipynb`. Deja la submission en `submissions/`.

## Convenciones del grupo

* **Folds fijos.** Todos los modelos se validan con `data/folds.csv` usando `run_cv` de `src/cv.py`. Así los AUC son comparables entre experimentos. Si alguien regenera los folds, tienen que dar idénticos (semilla fija); no se cambian sin avisar.
* **Preprocesamiento dentro del Pipeline**, para que se ajuste solo con el train de cada fold y no haya leakage.
* **Nombre de submissions:** `<fecha>_<modelo>_cv<AUC>.csv`, así se puede comparar el AUC de CV con el score de Kaggle.

## Bloque BUR_ (bureau + bureau_balance)

Reparto del grupo: cada integrante entrega un archivo con **una fila por `SK_ID_CURR`** y sus columnas prefijadas, y la union final es un merge por `SK_ID_CURR`. Este bloque es el de las tablas del Credit Bureau.

```
python -m src.features_bureau   # genera data/features_bureau.parquet (~30 s)
python -m src.eval_bureau       # CV con los folds fijos: application vs application + BUR_
```

* **Agregacion en dos niveles.** `bureau_balance` tiene una fila por credito y por mes (27,3M filas), asi que primero se resume el historial mensual de cada credito (`SK_ID_BUREAU`) y despues se agrega por cliente (`SK_ID_CURR`). El archivo se lee por chunks y los resumenes parciales se combinan con sumas, minimos y maximos.
* **78 features**: conteos de creditos (activos, cerrados, tipos, hipotecas, microcreditos), montos (otorgado, deuda, limite, mora, cuotas), tiempos (antiguedad, vencimientos, frescura del reporte), comportamiento de pago mes a mes (tasa de meses en mora, peor atraso, mora en los ultimos 12 meses), los mismos cortes **solo para los creditos vigentes** (sufijo `_ACT`) y ratios economicos (deuda/credito, mora/credito, creditos por anio).
* **`BUR_TIENE_BUREAU`** queda en NaN para los clientes sin historial en el bureau (14,3% de application), que es informacion en si misma.
* El parquet no se sube a git (se regenera con el comando de arriba).
