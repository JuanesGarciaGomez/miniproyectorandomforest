"""
train_model.py
==============
Script reproducible para entrenar y persistir el modelo de churn.

Hace exactamente lo mismo que el notebook `2_model_training.ipynb`, pero
en formato CLI para que CI/CD, contenedores Docker o pipelines Airflow
puedan invocarlo sin Jupyter.

Uso:
    python train_model.py                        # CPU, todos los modelos
    python train_model.py --use-gpu              # XGBoost en GPU
    python train_model.py --models xgb lgbm      # solo entrenar dos
    python train_model.py --quick                # GridSearch reducido (debug)

Salida:
    app/model.joblib  →  diccionario {"model": Pipeline, "name": str,
                                       "metrics": dict, "features": dict}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from app.preprocessing import (
    TARGET_COLUMN,
    build_feature_frame,
    build_preprocessor,
    clean_total_charges,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_OUT = ROOT / "app" / "model.joblib"
METRICS_OUT = ROOT / "app" / "metrics.json"

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Carga y preparación de datos
# ----------------------------------------------------------------------
def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"No encuentro el dataset en {path}. "
            f"Sigue las instrucciones en data/README.md."
        )

    df = pd.read_csv(path)

    # customerID no aporta señal: lo descartamos.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df = clean_total_charges(df)

    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = build_feature_frame(df.drop(columns=[TARGET_COLUMN]))

    logger.info("Dataset: %d filas × %d columnas. Tasa de churn: %.2f%%",
                len(df), df.shape[1], 100 * y.mean())
    return X, y


# ----------------------------------------------------------------------
# Definición de modelos y grids
# ----------------------------------------------------------------------
def get_model_grids(use_gpu: bool, quick: bool) -> dict[str, dict[str, Any]]:
    """Devuelve un dict {nombre: {estimator, param_grid}}.

    `quick=True` reduce el grid para iteraciones rápidas en debug.
    """
    grids: dict[str, dict[str, Any]] = {}

    # ---------- Random Forest ----------
    rf_grid = {
        "model__n_estimators": [200] if quick else [200, 400],
        "model__max_depth": [None, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
        "model__bootstrap": [True],
        "model__criterion": ["gini"] if quick else ["gini", "entropy"],
    }
    grids["random_forest"] = {
        "estimator": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
        ),
        "param_grid": rf_grid,
    }

    # ---------- XGBoost ----------
    try:
        from xgboost import XGBClassifier

        xgb_kwargs: dict[str, Any] = {
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "tree_method": "hist",
        }
        if use_gpu:
            # API moderna xgboost ≥ 2.0 — `gpu_hist` está deprecado.
            xgb_kwargs["device"] = "cuda"
            logger.info("XGBoost configurado para usar GPU (device='cuda').")

        xgb_grid = {
            "model__n_estimators": [200] if quick else [200, 400],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.1] if quick else [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__gamma": [0, 1],
            "model__min_child_weight": [1, 3],
            "model__reg_alpha": [0, 0.1],
            "model__reg_lambda": [1.0],
            # Para clases desbalanceadas. neg/pos ≈ 73/27 ≈ 2.7 en Telco.
            "model__scale_pos_weight": [1.0, 2.7],
        }
        grids["xgboost"] = {
            "estimator": XGBClassifier(**xgb_kwargs),
            "param_grid": xgb_grid,
        }
    except ImportError:
        logger.warning("xgboost no instalado, lo omito.")

    # ---------- CatBoost ----------
    try:
        from catboost import CatBoostClassifier

        cb_grid = {
            "model__iterations": [200] if quick else [200, 400],
            "model__depth": [4, 6],
            "model__learning_rate": [0.1] if quick else [0.05, 0.1],
            "model__l2_leaf_reg": [3, 5],
            "model__bagging_temperature": [0, 1],
            "model__border_count": [128],
            "model__random_strength": [1],
        }
        grids["catboost"] = {
            "estimator": CatBoostClassifier(
                random_seed=RANDOM_STATE, verbose=False,
            ),
            "param_grid": cb_grid,
        }
    except ImportError:
        logger.warning("catboost no instalado, lo omito.")

    # ---------- LightGBM ----------
    try:
        from lightgbm import LGBMClassifier

        lgbm_grid = {
            "model__n_estimators": [200] if quick else [200, 400],
            "model__max_depth": [-1, 6],
            "model__learning_rate": [0.1] if quick else [0.05, 0.1],
            "model__num_leaves": [31, 63],
            "model__min_child_samples": [20, 30],
            "model__min_child_weight": [1e-3],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__reg_alpha": [0, 0.1],
            "model__reg_lambda": [0, 0.1],
            "model__max_bin": [255],
        }
        grids["lightgbm"] = {
            "estimator": LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
                class_weight="balanced",
            ),
            "param_grid": lgbm_grid,
        }
    except ImportError:
        logger.warning("lightgbm no instalado, lo omito.")

    return grids


# ----------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------
def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def train_one(
    name: str,
    estimator: Any,
    param_grid: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> tuple[Pipeline, dict[str, float], dict[str, Any]]:
    """Crea pipeline preprocesador+modelo, ajusta GridSearchCV, evalúa."""
    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])

    logger.info("[%s] iniciando GridSearchCV (%d combinaciones aprox.)",
                name,
                int(np.prod([len(v) for v in param_grid.values()])))

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    metrics = evaluate(search.best_estimator_, X_test, y_test)
    logger.info("[%s] mejores params: %s", name, search.best_params_)
    logger.info("[%s] AUC test: %.4f | F1 test: %.4f", name, metrics["roc_auc"], metrics["f1"])

    return search.best_estimator_, metrics, search.best_params_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gpu", action="store_true",
                        help="Activa device='cuda' para XGBoost.")
    parser.add_argument("--quick", action="store_true",
                        help="Reduce drásticamente el grid (debug).")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["random_forest", "xgboost", "catboost", "lightgbm"],
                        help="Subset de modelos a entrenar.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_OUT)
    args = parser.parse_args()

    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grids = get_model_grids(use_gpu=args.use_gpu, quick=args.quick)
    if args.models:
        grids = {k: v for k, v in grids.items() if k in args.models}
        if not grids:
            raise SystemExit("Ningún modelo seleccionado disponible.")

    results: dict[str, dict[str, Any]] = {}
    best_pipeline: Pipeline | None = None
    best_name: str = ""
    best_auc: float = -1.0

    for name, conf in grids.items():
        pipe, metrics, params = train_one(
            name, conf["estimator"], conf["param_grid"],
            X_train, y_train, X_test, y_test, cv,
        )
        results[name] = {"metrics": metrics, "best_params": params}

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_pipeline = pipe
            best_name = name

    assert best_pipeline is not None
    logger.info("=" * 60)
    logger.info("Mejor modelo: %s (AUC=%.4f)", best_name, best_auc)
    logger.info("=" * 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best_pipeline, "name": best_name, "metrics": results[best_name]["metrics"]},
        args.output,
    )
    logger.info("Modelo guardado en %s", args.output)

    METRICS_OUT.write_text(json.dumps(results, indent=2))
    logger.info("Métricas guardadas en %s", METRICS_OUT)


if __name__ == "__main__":
    main()
