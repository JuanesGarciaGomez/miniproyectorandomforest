import app.api as api_module
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(dummy_model):
    api_module.model = __import__("joblib").load(dummy_model)
    with TestClient(api_module.app) as c:
        yield c


def _valid_payload():
    return {
        "SeniorCitizen": 1,
        "MonthlyCharges": 99.9,
        "TotalCharges": 499.5,
        "Contract": "Month-to-month",
        "PaymentMethod": "Bank transfer (automatic)",
    }


def test_predict_returns_probability_and_label(client):
    resp = client.post("/predict", json=_valid_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in {"Yes", "No"}


def test_predict_rejects_missing_field(client):
    bad = _valid_payload()
    bad.pop("MonthlyCharges")
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_high_risk_profile(client):
    high_risk = {
        "SeniorCitizen": 1,
        "MonthlyCharges": 99.9,
        "TotalCharges": 499.5,
        "Contract": "Month-to-month",
        "PaymentMethod": "Bank transfer (automatic)",
    }
    resp = client.post("/predict", json=high_risk)
    assert resp.status_code == 200
    assert resp.json()["prediction"] == "Yes"


def test_predict_low_risk_profile(client):
    low_risk = {
        "SeniorCitizen": 0,
        "MonthlyCharges": 25.0,
        "TotalCharges": 2000.0,
        "Contract": "Two year",
        "PaymentMethod": "Mailed check",
    }
    resp = client.post("/predict", json=low_risk)
    assert resp.status_code == 200
    assert resp.json()["prediction"] == "No"
