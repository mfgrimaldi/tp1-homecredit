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
submissions/      CSV generados (no se suben a git)
```

## Cómo arrancar

1. Instalar dependencias en tu entorno de Python: `pip install -r requirements.txt`
2. Bajar los datos de Kaggle y dejarlos en `data/raw/` (al menos `application_train.csv` y `application_test.csv`). Alternativa: dejarlos en la carpeta que contiene al repo; `src/config.py` los busca ahí si `data/raw/` está vacío.
3. Los folds ya están generados en `data/folds.csv`. Solo si hiciera falta regenerarlos (desde la raíz del repo): `python -m src.make_folds`
4. Correr `notebooks/01_baseline_logistica.ipynb`. Deja la submission en `submissions/`.

## Convenciones del grupo

* **Folds fijos.** Todos los modelos se validan con `data/folds.csv` usando `run_cv` de `src/cv.py`. Así los AUC son comparables entre experimentos. Si alguien regenera los folds, tienen que dar idénticos (semilla fija); no se cambian sin avisar.
* **Preprocesamiento dentro del Pipeline**, para que se ajuste solo con el train de cada fold y no haya leakage.
* **Nombre de submissions:** `<fecha>_<modelo>_cv<AUC>.csv`, así se puede comparar el AUC de CV con el score de Kaggle.
