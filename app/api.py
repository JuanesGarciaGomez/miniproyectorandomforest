from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "app/model.joblib")
model = None


@app.on_event("startup")
def load_model():
    global model, MODEL_PATH
    MODEL_PATH = os.getenv("MODEL_PATH", "app/model.joblib")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None


class CustomerData(BaseModel):
    SeniorCitizen: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str


@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    }
    payment_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3,
    }

    features = np.array([[
        data.SeniorCitizen,
        data.MonthlyCharges,
        data.TotalCharges,
        contract_map.get(data.Contract, 0),
        payment_map.get(data.PaymentMethod, 0),
    ]])

    churn_proba = float(model.predict_proba(features)[0][1])

    return {
        "churn_probability": round(churn_proba, 2),
        "prediction": "Yes" if churn_proba > 0.5 else "No",
    }
