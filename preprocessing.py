"""
preprocessing.py
================
Lógica de preprocesamiento *compartida* entre el entrenamiento (notebooks /
`train_model.py`) y el servicio de inferencia (`api.py`). Mantenerla en un
único módulo evita el "training-serving skew": que el modelo en producción
reciba features en un formato distinto al que vio durante el entrenamiento.

El pipeline real de transformación (imputación, escalado, OHE) vive dentro
de un `sklearn.compose.ColumnTransformer` que se persiste junto con el
modelo en `model.joblib`. Aquí solo definimos:

- Las listas de columnas numéricas y categóricas.
- Una función `build_feature_frame` que toma datos crudos (dict o DataFrame)
  y los devuelve como un `pd.DataFrame` con las columnas exactas y en el
  orden esperado.
- Un `build_preprocessor()` que arma el ColumnTransformer.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------------------------------------------------
# Definición de columnas — fuente única de verdad
# ----------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",  # ya viene 0/1, lo tratamos como numérico
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "Churn"


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """`TotalCharges` viene como string en el CSV original y contiene
    espacios en blanco para clientes nuevos (tenure==0). Los convertimos
    a NaN para que el imputador del pipeline los maneje. Devuelve una
    copia, no modifica el DataFrame original.
    """
    df = df.copy()
    # Si no es numérica (típicamente object/str porque tiene espacios en blanco),
    # forzamos conversión: los valores no convertibles se vuelven NaN.
    if not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def build_feature_frame(payload: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Construye un DataFrame con exactamente las columnas que el pipeline
    espera, en el orden correcto. Sirve tanto para una sola observación
    (dict) como para un lote (DataFrame).

    Esto es lo que la API llama antes de invocar `model.predict_proba`.
    """
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    else:
        df = payload.copy()

    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    return df[ALL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Crea el ColumnTransformer.

    - Numéricas: imputar mediana + estandarizar.
    - Categóricas: imputar 'missing' + One-Hot Encoding (handle_unknown='ignore'
      para que en producción no truene si llega una categoría no vista).

    El escalado en realidad no es indispensable para los modelos basados en
    árboles, pero lo dejamos por consistencia y por si en el futuro
    comparas contra un modelo lineal.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor
