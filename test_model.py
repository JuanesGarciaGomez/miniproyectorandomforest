"""
test_model.py
=============
Pruebas del módulo de preprocesamiento y del modelo persistido.

Estos tests no requieren ejecutar el entrenamiento completo: para validar
la lógica del pipeline, entrenamos un modelo dummy mínimo en memoria con
unas pocas filas. Eso nos permite verificar:
- El ColumnTransformer encaja columnas correctamente.
- La función build_feature_frame respeta el orden y detecta faltantes.
- Un Pipeline preprocessor+modelo puede serializarse con joblib.
"""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from app.preprocessing import (
    ALL_FEATURES,
    build_feature_frame,
    build_preprocessor,
    clean_total_charges,
)


# ---------- Datos sintéticos ----------
def _sample_row() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }


# ---------- Tests de preprocesamiento ----------
def test_build_feature_frame_orders_columns():
    df = build_feature_frame(_sample_row())
    assert list(df.columns) == ALL_FEATURES
    assert len(df) == 1


def test_build_feature_frame_raises_on_missing():
    bad = _sample_row()
    bad.pop("tenure")
    with pytest.raises(ValueError, match="Faltan columnas"):
        build_feature_frame(bad)


def test_clean_total_charges_handles_blank_strings():
    df = pd.DataFrame({"TotalCharges": ["100.5", " ", "200"]})
    cleaned = clean_total_charges(df)
    assert cleaned["TotalCharges"].isna().sum() == 1
    assert cleaned["TotalCharges"].dtype.kind == "f"


# ---------- Test end-to-end del pipeline ----------
def _build_dummy_dataset(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """Genera un mini-dataset alternando dos perfiles para que el modelo
    pueda aprender algo trivial.
    """
    base = _sample_row()
    rows = []
    labels = []
    for i in range(n):
        row = base.copy()
        if i % 2 == 0:
            row["Contract"] = "Month-to-month"
            row["tenure"] = 1
            row["MonthlyCharges"] = 90.0
            labels.append(1)
        else:
            row["Contract"] = "Two year"
            row["tenure"] = 60
            row["MonthlyCharges"] = 30.0
            labels.append(0)
        rows.append(row)
    return pd.DataFrame(rows), pd.Series(labels)


def test_pipeline_can_fit_and_predict_proba():
    import numpy as np
    X, y = _build_dummy_dataset()
    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=20, random_state=0)),
    ])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    # Las dos columnas suman 1 (propiedad de probabilidades).
    assert np.all(np.abs(proba.sum(axis=1) - 1.0) < 1e-6)


def test_pipeline_serialization_roundtrip(tmp_path):
    X, y = _build_dummy_dataset()
    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
    ])
    pipe.fit(X, y)

    artifact_path = tmp_path / "model.joblib"
    joblib.dump({"model": pipe, "name": "rf_dummy"}, artifact_path)

    loaded = joblib.load(artifact_path)
    assert "model" in loaded
    assert loaded["name"] == "rf_dummy"
    proba = loaded["model"].predict_proba(X)
    assert proba.shape == (len(X), 2)
