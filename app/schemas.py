"""
schemas.py
==========
Modelos Pydantic v2 que definen el contrato de entrada y salida de la API
de inferencia de churn. Esto sirve como:

1. Validación automática del JSON entrante (FastAPI lo enlaza al endpoint).
2. Documentación: Swagger en /docs muestra el esquema de ejemplo.
3. Punto único de verdad sobre las columnas que el modelo espera.

Si en el futuro cambias el set de features durante el entrenamiento, este
archivo y `preprocessing.build_feature_frame` deben actualizarse en conjunto.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """Datos crudos de un cliente que la API recibe para predecir churn.

    Las opciones permitidas (Literal[...]) coinciden exactamente con los
    valores que aparecen en el dataset Telco. Si un cliente envía algo
    distinto, FastAPI devuelve 422 antes de tocar el modelo.
    """

    # ---- Demográficas ----
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    # ---- Cuenta y contrato ----
    tenure: int = Field(..., ge=0, le=100, description="Meses como cliente")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    # ---- Cargos ----
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Respuesta del endpoint /predict."""

    churn_probability: float = Field(..., ge=0.0, le=1.0,
                                     description="Probabilidad de que el cliente abandone (clase 1).")
    prediction: Literal["Yes", "No"] = Field(..., description="Decisión binaria con umbral 0.5.")
    model_name: str = Field(..., description="Nombre del modelo que produjo la predicción.")


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""

    status: Literal["ok", "error"]
    model_loaded: bool
    model_name: str | None = None
