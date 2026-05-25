from contextlib import asynccontextmanager
import os

import joblib
from fastapi import FastAPI, HTTPException

from app.schemas import CustomerData, HealthResponse, PredictionResponse
from app.preprocessing import build_feature_frame

_artifact = {}


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_path = os.getenv("MODEL_PATH", "app/model.joblib")
    try:
        _artifact.update(joblib.load(model_path))
    except Exception:
        pass
    yield
    _artifact.clear()


app = FastAPI(
    title="Telco Churn Predictor",
    description="API de inferencia para predecir churn de clientes Telco.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = "model" in _artifact
    return HealthResponse(
        status="ok" if loaded else "error",
        model_loaded=loaded,
        model_name=_artifact.get("name") if loaded else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if "model" not in _artifact:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    pipeline = _artifact["model"]
    X = build_feature_frame(data.model_dump())
    proba = float(pipeline.predict_proba(X)[0][1])

    return PredictionResponse(
        churn_probability=round(proba, 4),
        prediction="Yes" if proba > 0.5 else "No",
        model_name=_artifact["name"],
    )
