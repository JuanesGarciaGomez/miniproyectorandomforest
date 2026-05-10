"""
test_api.py
===========
Tests de la API FastAPI usando TestClient.

Para no depender del modelo real (que tardaría mucho en entrenarse en CI),
generamos un modelo *dummy* en una fixture, lo guardamos en una ruta
temporal y apuntamos `MODEL_PATH` allí antes de importar la app.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from app.preprocessing import build_preprocessor


def _build_and_dump_dummy(model_path: Path) -> None:
    """Entrena un RF mínimo con datos sintéticos y lo persiste."""
    rows = []
    labels = []
    base = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0, "TotalCharges": 70.0,
    }
    for i in range(40):
        r = base.copy()
        if i % 2 == 0:
            r["Contract"] = "Month-to-month"
            r["tenure"] = 1
            labels.append(1)
        else:
            r["Contract"] = "Two year"
            r["tenure"] = 60
            labels.append(0)
        rows.append(r)

    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
    ])
    pipe.fit(pd.DataFrame(rows), labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "name": "rf_dummy"}, model_path)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Crea TestClient con un modelo dummy en MODEL_PATH."""
    tmp_dir = tmp_path_factory.mktemp("model_artifacts")
    model_path = tmp_dir / "model.joblib"
    _build_and_dump_dummy(model_path)

    os.environ["MODEL_PATH"] = str(model_path)

    # Importamos AQUÍ (no arriba) para que la variable de entorno ya esté
    # seteada cuando el módulo se evalúe.
    from fastapi.testclient import TestClient
    from app import api as api_module

    # Forzamos recarga del modelo (por si el módulo ya estaba cacheado).
    api_module._model = None
    api_module._model_name = "unknown"
    api_module.MODEL_PATH = model_path
    api_module._load_model()

    with TestClient(api_module.app) as c:
        yield c


def _valid_payload() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


# ---------- Tests ----------
def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "service" in body


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "rf_dummy"


def test_predict_returns_probability_and_label(client):
    resp = client.post("/predict", json=_valid_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in {"Yes", "No"}
    assert body["model_name"] == "rf_dummy"


def test_predict_rejects_invalid_category(client):
    bad = _valid_payload()
    bad["Contract"] = "Forever"   # no está en el Literal
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    bad = _valid_payload()
    bad.pop("MonthlyCharges")
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_negative_tenure(client):
    bad = _valid_payload()
    bad["tenure"] = -3
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
