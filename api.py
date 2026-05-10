"""
api.py
======
Servicio REST para servir el modelo de predicción de churn.

Endpoints:
- GET  /            → mensaje de bienvenida.
- GET  /health      → liveness/readiness check (útil para Docker, k8s, CI).
- POST /predict     → recibe un cliente (CustomerData) y devuelve la
                      probabilidad de churn + decisión binaria.

Levantar localmente:
    uvicorn app.api:app --reload --port 8000

Documentación interactiva (Swagger):
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException

from app.preprocessing import build_feature_frame
from app.schemas import CustomerData, HealthResponse, PredictionResponse

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("telco-churn-api")

# La ruta del modelo se puede sobrescribir con la variable de entorno
# MODEL_PATH (útil en tests para cargar un modelo dummy).
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

# ----------------------------------------------------------------------
# Estado y carga del modelo
# ----------------------------------------------------------------------
_model: Any | None = None
_model_name: str = "unknown"


def _load_model() -> None:
    """Carga el modelo desde disco si aún no está en memoria.

    Si falla, dejamos `_model = None` y el endpoint /predict responderá
    503. Esto evita que la app crashee al iniciar si el artefacto aún
    no se ha generado.
    """
    global _model, _model_name

    if _model is not None:
        return

    if not MODEL_PATH.exists():
        logger.error("No se encontró el modelo en %s", MODEL_PATH)
        return

    try:
        artifact = joblib.load(MODEL_PATH)
        # `artifact` puede ser un Pipeline directo o un dict con metadata.
        if isinstance(artifact, dict) and "model" in artifact:
            _model = artifact["model"]
            _model_name = artifact.get("name", "unknown")
        else:
            _model = artifact
            _model_name = type(artifact).__name__

        logger.info("Modelo '%s' cargado desde %s", _model_name, MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensivo
        logger.exception("Error cargando el modelo: %s", exc)
        _model = None


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carga el modelo al arrancar la app (reemplaza el viejo on_event)."""
    _load_model()
    yield
    # Aquí podrían ir tareas de cleanup al apagar la app.


app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "Servicio de inferencia para predecir abandono (churn) de clientes "
        "de una empresa de telecomunicaciones, basado en modelos de árboles "
        "(Random Forest / XGBoost / CatBoost / LightGBM)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Telco Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health-check sin efectos secundarios."""
    return HealthResponse(
        status="ok" if _model is not None else "error",
        model_loaded=_model is not None,
        model_name=_model_name if _model is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData) -> PredictionResponse:
    """Devuelve la probabilidad de churn para un cliente.

    Pydantic ya validó los tipos y dominios de cada campo antes de
    llegar aquí, así que solo nos toca:
    1. Convertir el payload en DataFrame con las columnas correctas.
    2. Llamar al pipeline (preprocesamiento + modelo).
    3. Devolver probabilidad + decisión binaria.
    """
    if _model is None:
        # Intento perezoso: por si MODEL_PATH cambió en runtime (tests).
        _load_model()

    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible. Verifica que exista {MODEL_PATH}.",
        )

    try:
        df = build_feature_frame(data.model_dump())
        proba = float(_model.predict_proba(df)[0][1])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensivo
        logger.exception("Error durante la inferencia")
        raise HTTPException(status_code=500, detail="Error de inferencia.") from exc

    decision = "Yes" if proba > PREDICTION_THRESHOLD else "No"

    return PredictionResponse(
        churn_probability=round(proba, 4),
        prediction=decision,
        model_name=_model_name,
    )
