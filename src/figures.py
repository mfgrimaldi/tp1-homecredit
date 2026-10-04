"""Gráficos para el informe de la Parte 1. Se guardan en resultados/figuras/.

Paleta accesible (validada para daltonismo): azul y naranja como series,
diverging azul-gris-rojo para valores bajos/altos de una variable.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import ROOT

FIG_DIR = ROOT / "resultados" / "figuras"

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
SURFACE, INK, INK_2, GRID, MID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c3"
DIVERGING = LinearSegmentedColormap.from_list("bajo_alto", [BLUE, "#f0efec", RED])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})

# Variables creadas en src/features.py (para distinguirlas de las originales)
ENGINEERED_PREFIXES = ("EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_MAX", "EXT_SOURCE_STD",
                       "EXT_SOURCE_PROD", "CREDIT_INCOME", "ANNUITY_INCOME", "CREDIT_TERM",
                       "GOODS_CREDIT", "INCOME_PER_PERSON", "AGE_YEARS", "EMPLOYED_AGE",
                       "DAYS_EMPLOYED_ANOM", "DOCS_COUNT", "BUREAU_", "PREV_", "INST_")


def is_engineered(col: str) -> bool:
    return col.startswith(ENGINEERED_PREFIXES)


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def auc_evolution(steps: pd.DataFrame) -> Path:
    """Dot plot del AUC por etapa. steps: columnas etapa, cv, kaggle (NaN si no se subió)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    yy = np.arange(len(steps))[::-1]
    ax.scatter(steps["cv"], yy, s=70, color=BLUE, zorder=3, label="Validación cruzada (5 folds)",
               edgecolor=SURFACE, linewidth=2)
    k = steps["kaggle"].notna()
    ax.scatter(steps.loc[k, "kaggle"], yy[k.to_numpy()], s=70, color=ORANGE, marker="D", zorder=3,
               label="Kaggle (Private)", edgecolor=SURFACE, linewidth=2)
    for y_, v in zip(yy, steps["cv"]):
        ax.annotate(f"{v:.4f}", (v, y_), xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=INK_2)
    ax.set_yticks(yy, steps["etapa"])
    ax.set_xlabel("ROC AUC")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.725, 0.795)
    ax.set_title("Cada etapa sube el AUC; Kaggle confirma la validación")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, "01_evolucion_auc")


def optuna_history(trials: pd.DataFrame) -> Path:
    """AUC de cada prueba de Optuna y el mejor acumulado."""
    t = trials[trials["state"] == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.scatter(t["number"], t["value"], s=40, color=BLUE, label="AUC de cada combinación",
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.step(t["number"], t["value"].cummax(), where="post", color=ORANGE, linewidth=2,
            label="Mejor hasta el momento")
    ax.set_xlabel("Número de prueba")
    ax.set_ylabel("AUC (CV, 5 folds)")
    ax.set_title("Búsqueda de hiperparámetros con Optuna (30 pruebas)")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "02_optuna_historia")


def importance_bars(shap_imp: pd.Series, top: int = 15) -> Path:
    """Importancia global (media de |SHAP|), coloreando variables creadas vs. originales."""
    s = shap_imp.head(top)[::-1]
    colors = [BLUE if is_engineered(c) else ORANGE for c in s.index]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(s.index, s.values, color=colors, height=0.7, edgecolor=SURFACE, linewidth=2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Impacto medio en la predicción (media de |SHAP|, log-odds)")
    ax.set_title(f"Las {top} variables más importantes del LightGBM")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["Creada en feature engineering", "Original de Kaggle"],
              loc="lower right", fontsize=9)
    return _save(fig, "03_importancia_shap")


def shap_beeswarm(shap_df: pd.DataFrame, X: pd.DataFrame, top: int = 15) -> Path:
    """Beeswarm: cada punto es un cliente; posición = efecto, color = valor de la variable."""
    import shap
    cols = shap_df.abs().mean().sort_values(ascending=False).head(top).index
    Xn = X[cols].copy()
    for c in Xn.columns:  # las categóricas se muestran por código (el color no aplica)
        if not pd.api.types.is_numeric_dtype(Xn[c]):
            Xn[c] = Xn[c].astype("category").cat.codes.replace(-1, np.nan)
    shap.summary_plot(shap_df[cols].to_numpy(), Xn, max_display=top, cmap=DIVERGING,
                      show=False, plot_size=(8, 6), color_bar_label="Valor de la variable")
    fig = plt.gcf()
    fig.axes[-1].set_yticklabels(["Bajo", "Alto"])  # barra de color (shap la deja en inglés)
    fig.axes[0].set_xlabel("Efecto en la predicción (SHAP, log-odds): + sube el riesgo, − lo baja")
    fig.axes[0].set_title("Cómo empuja cada variable la predicción de riesgo", loc="left")
    return _save(fig, "04_shap_beeswarm")


def shap_dependence(shap_df: pd.DataFrame, X: pd.DataFrame, cols: list[str], labels: list[str]) -> Path:
    """Paneles valor de la variable vs. su efecto SHAP."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, c, lab in zip(axes.ravel(), cols, labels):
        x = X[c]
        hi = x.quantile(0.99)
        ok = x.notna() & (x <= hi)
        ax.scatter(x[ok], shap_df.loc[ok, c], s=4, alpha=0.25, color=BLUE, linewidths=0)
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel(c, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("SHAP (+ más riesgo)")
    fig.suptitle("Efecto de cada variable según su valor", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_shap_dependencia")
