# Telco Customer Churn — Proyecto MLOps

Predicción de abandono (churn) de clientes en una empresa de telecomunicaciones, usando árboles de decisión ensamblados (Random Forest, XGBoost, CatBoost, LightGBM), interpretabilidad con LIME, y un servicio de inferencia FastAPI desplegable con Docker y CI/CD en GitHub Actions.

---

## Tabla de contenidos

1. [Descripción](#descripción)
2. [Arquitectura](#arquitectura)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Instalación](#instalación)
5. [Uso — Notebooks](#uso--notebooks)
6. [Uso — Script de entrenamiento](#uso--script-de-entrenamiento)
7. [Uso — API local](#uso--api-local)
8. [Uso — Docker](#uso--docker)
9. [Tests](#tests)
10. [CI/CD](#cicd)
11. [Recomendaciones de MLOps](#recomendaciones-de-mlops)

---

## Descripción

El dataset **Telco Customer Churn** (Kaggle, ~7043 filas) contiene información demográfica, de servicios contratados y facturación de clientes, junto con la etiqueta `Churn` (Yes/No). El churn ronda el 27%, así que es un problema de **clasificación binaria con desbalance moderado**.

El proyecto cubre:

- **EDA** y preprocesamiento documentado (notebook 1).
- **4 modelos** entrenados con `Pipeline` + `GridSearchCV` + `StratifiedKFold(5)`, con XGBoost en GPU si está disponible (notebook 2).
- **Interpretabilidad local** con LIME sobre 3 casos representativos (notebook 3).
- **Servicio REST** con FastAPI (`app/api.py`) que carga el mejor pipeline y expone `/predict`.
- **Contenerización** con Dockerfile multi-stage minimalista.
- **CI/CD** con GitHub Actions (lint + tests + build de imagen).
- **Tests unitarios** del módulo de preprocesamiento y de la API con `TestClient`.

---

## Arquitectura

```
                                     ┌──────────────────────────┐
                                     │    notebooks/2_model     │
   data/telco_churn.csv  ─────────►  │    train_model.py        │
   (raw, no en repo)                 └──────────┬───────────────┘
                                                │ joblib.dump
                                                ▼
                                        app/model.joblib
                                                │
                                                │ joblib.load
                                                ▼
   POST /predict  ──►  FastAPI (Uvicorn)  ──►  Pipeline
                       │                       (preprocessor + clf)
                       │                       │
                       └─►  Pydantic            └─►  predict_proba
                            validation                ▼
                                            JSON {churn_probability,
                                                  prediction, model_name}
```

Todo el código que el modelo necesita en serving (definición de columnas, ColumnTransformer) está en **`app/preprocessing.py`** y se importa tanto en entrenamiento como en la API → no hay training-serving skew.

---

## Estructura del repositorio

```
telco-churn-mlops/
├── data/
│   ├── README.md                  # Cómo descargar el dataset
│   └── telco_churn.csv            # (no incluido — descárgalo de Kaggle)
├── notebooks/
│   ├── 1_eda_preprocessing.ipynb  # EDA + decisiones de limpieza
│   ├── 2_model_training.ipynb     # Pipelines, GridSearchCV, evaluación
│   └── 3_interpretability.ipynb   # LIME en 3 casos
├── app/
│   ├── __init__.py
│   ├── api.py                     # FastAPI service
│   ├── schemas.py                 # Pydantic models (input/output)
│   ├── preprocessing.py           # Lógica compartida train+serving
│   └── model.joblib               # (se genera al entrenar)
├── tests/
│   ├── __init__.py
│   ├── test_model.py              # Tests del preprocessing y pipeline
│   └── test_api.py                # Tests de la API con TestClient
├── train_model.py                 # CLI reproducible de entrenamiento
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .github/workflows/ci.yml       # Pipeline GitHub Actions
├── .gitignore
└── README.md
```

---

## Instalación

### Requisitos
- Python ≥ 3.10
- (Opcional) GPU NVIDIA + drivers CUDA para acelerar XGBoost.
- Docker ≥ 24 (para contenerización).

### Setup local

```bash
# Clonar
git clone <tu-repo>
cd telco-churn-mlops

# Entorno virtual
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows PowerShell

# Dependencias
pip install --upgrade pip
pip install -r requirements.txt

# Descargar el dataset (ver data/README.md) y dejarlo en data/telco_churn.csv
```

---

## Uso — Notebooks

```bash
jupyter lab        # o abre los .ipynb directamente desde VS Code
```

Ejecuta en orden:
1. `1_eda_preprocessing.ipynb` — entender los datos.
2. `2_model_training.ipynb` — entrenar y persistir `app/model.joblib`.
3. `3_interpretability.ipynb` — explicar predicciones con LIME.

> Si tienes GPU NVIDIA, en el notebook 2 deja `USE_GPU = True`. La API moderna de XGBoost ≥ 2.0 usa `tree_method='hist'` + `device='cuda'`. **No uses `gpu_hist`**: está deprecado.

---

## Uso — Script de entrenamiento

Para reproducir el entrenamiento sin Jupyter (ej. desde CI o un cron):

```bash
# Entrenamiento completo, todos los modelos, CPU
python train_model.py

# Con GPU para XGBoost
python train_model.py --use-gpu

# Subset de modelos
python train_model.py --models xgboost lightgbm --use-gpu

# Modo "quick" para iterar rápido (grid muy reducido)
python train_model.py --quick
```

Outputs:
- `app/model.joblib` — mejor pipeline (preprocesador + modelo) según AUC test.
- `app/metrics.json` — métricas de los 4 modelos para comparación.

---

## Uso — API local

```bash
# Asegúrate de haber generado app/model.joblib antes
uvicorn app.api:app --reload --port 8000
```

Documentación interactiva (Swagger UI): http://localhost:8000/docs

### Endpoints

| Método | Ruta      | Descripción                                  |
|--------|-----------|----------------------------------------------|
| GET    | `/`       | Mensaje de bienvenida.                       |
| GET    | `/health` | Estado del servicio + si el modelo cargó.    |
| POST   | `/predict`| Probabilidad de churn para un cliente.       |

### Ejemplo `curl`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 29.85
  }'
```

Respuesta:
```json
{
  "churn_probability": 0.6231,
  "prediction": "Yes",
  "model_name": "xgboost"
}
```

---

## Uso — Docker

### Build

```bash
docker build -t telco-churn-api .
```

> ⚠️ Antes de construir la imagen, debes haber entrenado el modelo (`app/model.joblib` debe existir): el `Dockerfile` lo copia tal cual. Esto es deliberado — separa entrenamiento (con GPU, en tu host) de serving (CPU, en el contenedor).

### Ejecución (modo persistente)

```bash
# Crear contenedor con nombre fijo (solo la primera vez)
docker run -d --name telco-churn-container -p 8000:8000 telco-churn-api

# Después puedes pararlo / arrancarlo
docker stop  telco-churn-container
docker start telco-churn-container

# Logs
docker logs -f telco-churn-container

# Probar
curl http://localhost:8000/health
```

### Persistencia y volúmenes

Si quisieras montar un modelo entrenado externamente (sin reconstruir la imagen):

```bash
docker run -d --name telco-churn-container \
  -p 8000:8000 \
  -v $(pwd)/app/model.joblib:/app/app/model.joblib:ro \
  -e MODEL_PATH=/app/app/model.joblib \
  telco-churn-api
```

---

## Tests

```bash
# Desde la raíz del proyecto
pytest tests/ -v
```

Cubren:
- `tests/test_model.py`: `build_feature_frame`, manejo de `TotalCharges` con espacios, fit + predict_proba del pipeline, roundtrip de joblib.
- `tests/test_api.py`: endpoints `/`, `/health`, `/predict` con `TestClient`. Validación Pydantic (categorías inválidas, campos faltantes, tenure negativo).

Los tests **no requieren el dataset real**: entrenan un modelo dummy en memoria con datos sintéticos.

---

## CI/CD

`.github/workflows/ci.yml` define un pipeline que se dispara en cada push y pull request a `main`:

1. **Lint** con flake8 (configuración relajada en E501/W503).
2. **Tests** con pytest.
3. **Build Docker** (sin push) si los tests pasan y estamos en `main`.

Para publicar la imagen a Docker Hub, descomenta el job `publish` y configura los secrets `DOCKERHUB_USERNAME` y `DOCKERHUB_TOKEN`.

---

## Recomendaciones de MLOps

Lo que está incluido es un MVP. Para un sistema productivo de verdad, los siguientes pasos son razonables:

### 1. Monitoreo en producción
- **Logging estructurado** de cada predicción (input, output, latencia, modelo) → Elastic / Datadog / Cloudwatch.
- **Métricas Prometheus** vía `prometheus-fastapi-instrumentator`: requests/seg, latencia P95, error rate.
- **Alertas** sobre tasas de error, latencia, o caídas del `/health`.

### 2. Detección de deriva (drift)
- **Data drift**: monitorear distribuciones de las features de input vs. distribución de entrenamiento. Herramientas: `evidently`, `whylogs`, `nannyml`.
- **Concept drift**: cuando se conoce el outcome (churn real a los X meses), comparar predicciones vs. realidad. Si el AUC en producción baja > 5% del baseline, disparar reentrenamiento.

### 3. Reentrenamiento programado
- Cron job o Airflow DAG que:
  1. Extrae datos frescos del data warehouse.
  2. Ejecuta `train_model.py`.
  3. Compara métricas vs. modelo en producción.
  4. Si mejora, despliega; si no, alerta y mantiene el actual.
- **Champion/Challenger**: mantener dos modelos en paralelo, enrutando 5–10% del tráfico al challenger.

### 4. Despliegue en la nube
- **Opción simple**: AWS App Runner, Google Cloud Run, Azure Container Apps — toman tu imagen Docker y la escalan automáticamente. Coste bajo, latencia decente.
- **Opción robusta**: EKS/GKE/AKS con HPA (Horizontal Pod Autoscaler) sobre métricas de CPU o requests/seg. Configura readinessProbe apuntando a `/health`.
- **Opción serverless**: AWS Lambda con contenedores (límite de tamaño 10 GB) para tráfico esporádico — paga solo por inferencia.

### 5. Versionado y registry
- **MLflow Tracking** para registrar cada experimento (hiperparámetros, métricas, artefactos).
- **MLflow Model Registry** para versionar `model.joblib` con etapas `Staging → Production → Archived`.
- **DVC** para versionar el dataset junto con el código.

### 6. Escalabilidad
- Para latencias < 50ms y tráfico alto, considera serializar el pipeline a **ONNX** y servirlo con Triton Inference Server o ONNX Runtime — ganancias de 2–10× en latencia.
- Inferencia por lotes (batch endpoints) para reportes diarios, en vez de llamadas individuales.

### 7. Seguridad
- API behind un API Gateway con autenticación (API key, OAuth2).
- Rate limiting por cliente.
- HTTPS obligatorio en producción.
- Secrets (DB credentials, API keys) en un secret manager, nunca en imágenes Docker.

---

## Licencia y referencias

- Dataset: [Telco Customer Churn (Kaggle, BlastChar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- Bibliografía base: capítulo "Random Forest y XGBoost" de Lihki Rubio.

> ⚠️ Este proyecto es educativo. Los modelos y umbrales no están calibrados para uso comercial real sin validación adicional, calibración de probabilidades y pruebas A/B.
