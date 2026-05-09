# Telco Customer Churn — Mini Proyecto Random Forest

Predicción de abandono (churn) de clientes en una empresa de telecomunicaciones usando Random Forest, servido como API REST con FastAPI, contenerizado con Docker y con integración continua mediante GitHub Actions.

---

## Tabla de contenidos

1. [Descripción](#descripción)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Instalación](#instalación)
4. [Uso — API local](#uso--api-local)
5. [Uso — Docker](#uso--docker)
6. [Tests](#tests)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Flujo del Pipeline de CI](#flujo-del-pipeline-de-ci)

---

## Descripción

El dataset **Telco Customer Churn** contiene información de clientes de una empresa de telecomunicaciones con la etiqueta `Churn` (Yes/No). Este proyecto implementa:

- **Análisis exploratorio** y preprocesamiento (notebook 1).
- **Entrenamiento** de un modelo Random Forest con scikit-learn (notebook 2).
- **Interpretabilidad** con LIME (notebook 3).
- **Servicio REST** con FastAPI (`app/api.py`) que expone el endpoint `/predict`.
- **Contenerización** con Docker usando imagen `python:3.10-slim`.
- **CI/CD** con GitHub Actions (lint + tests automáticos).

---

## Estructura del repositorio

```
miniproyectorandomforest/
├── app/
│   ├── __init__.py
│   ├── api.py              # Servicio FastAPI con endpoint /predict
│   └── model.joblib        # Modelo entrenado (se genera al entrenar)
├── data/
│   └── README.md           # Instrucciones para descargar el dataset
├── notebooks/
│   ├── 1_eda_preprocessing.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_interpretability.ipynb
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Fixture con modelo dummy para CI
│   └── test_api.py         # Tests del endpoint /predict
├── .github/
│   └── workflows/
│       └── ci.yml          # Pipeline de integración continua
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Instalación

### Requisitos

- Python 3.10+
- Docker (para contenerización)

### Setup local

```bash
# Clonar el repositorio
git clone https://github.com/JuanesGarciaGomez/miniproyectorandomforest
cd miniproyectorandomforest

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso — API local

### Paso 1: Entrenar el modelo

Ejecuta el notebook `notebooks/2_model_training.ipynb` para generar `app/model.joblib`.

### Paso 2: Levantar el servidor

```bash
uvicorn app.api:app --port 8001
```

### Paso 3: Hacer una predicción

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"SeniorCitizen": 1, "MonthlyCharges": 99.9, "TotalCharges": 499.5, "Contract": "Month-to-month", "PaymentMethod": "Bank transfer (automatic)"}'
```

**Salida esperada:**

```json
{"churn_probability": 0.96, "prediction": "Yes"}
```

### Esquema de entrada (`CustomerData`)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `SeniorCitizen` | `int` | 1 si el cliente es adulto mayor, 0 si no |
| `MonthlyCharges` | `float` | Cargo mensual del cliente |
| `TotalCharges` | `float` | Cargo total acumulado |
| `Contract` | `str` | Tipo de contrato: `"Month-to-month"`, `"One year"`, `"Two year"` |
| `PaymentMethod` | `str` | Método de pago: `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |

### Esquema de salida

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `churn_probability` | `float` | Probabilidad de que el cliente abandone (entre 0 y 1) |
| `prediction` | `str` | `"Yes"` si la probabilidad > 0.5, `"No"` en caso contrario |

---

## Uso — Docker

### Construir y ejecutar (primera vez)

> Asegúrate de tener `app/model.joblib` antes de construir la imagen.

```bash
docker build -t telco-churn . && docker run -d --name telco-churn-container -p 8000:8000 telco-churn
```

### Gestionar el contenedor

```bash
# Iniciar
docker start telco-churn-container

# Detener
docker stop telco-churn-container

# Ver estado
docker ps -a | grep telco-churn
```

### Probar desde Docker

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"SeniorCitizen": 1, "MonthlyCharges": 99.9, "TotalCharges": 499.5, "Contract": "Month-to-month", "PaymentMethod": "Bank transfer (automatic)"}'
```

**Salida esperada:**

```json
{"churn_probability": 0.96, "prediction": "Yes"}
```

---

## Tests

```bash
pytest tests/ -v
```

Los tests utilizan un modelo dummy entrenado con datos sintéticos, por lo que **no requieren el dataset real** ni el modelo entrenado. Son ideales para ejecutarse en CI/CD.

---

## CI/CD Pipeline

El archivo `.github/workflows/ci.yml` define el pipeline de integración continua.

**Se ejecuta automáticamente** en cada `push` o `pull request` al repositorio.

### Pasos del pipeline

```
Push / Pull Request
        │
        ▼
1. Clonar el repositorio (actions/checkout)
        │
        ▼
2. Configurar Python 3.10
        │
        ▼
3. Instalar dependencias (pip install -r requirements.txt)
        │
        ▼
4. Verificar calidad de código (flake8 app/)
        │
    ┌───┴───┐
    │ Error │ → Pipeline falla. Corregir antes de continuar.
    └───────┘
        │ OK
        ▼
5. Ejecutar tests automáticos (pytest tests/)
        │
    ┌───┴───┐
    │ Error │ → Pipeline falla. Corregir y volver a intentar.
    └───────┘
        │ OK
        ▼
6. Pipeline exitoso ✓ — Código listo para revisión o despliegue
```

### Cómo ver los resultados

1. Ve a tu repositorio en GitHub.
2. Haz clic en la pestaña **Actions**.
3. Verás la lista de ejecuciones con su estado (✓ éxito o ✗ fallo).
4. Haz clic en cualquier ejecución para ver los detalles paso a paso.

---

## Flujo del Pipeline de Integración Continua (CI)

| Paso | Descripción |
|------|-------------|
| **1. Inicio automático** | El pipeline se ejecuta en cada `push` o `pull request` |
| **2. Clonación** | El código se descarga en un entorno limpio de Ubuntu |
| **3. Configuración Python** | Se instala Python 3.10 para garantizar compatibilidad |
| **4. Instalación de dependencias** | Se ejecuta `pip install -r requirements.txt` |
| **5. Verificación de estilo** | `flake8 app/ --ignore=E501,W503` revisa convenciones de código |
| **6. Detección de errores** | Si hay errores de estilo, el pipeline se detiene |
| **7. Pruebas automáticas** | `pytest tests/` valida la funcionalidad del endpoint |
| **8. Resultado** | ✓ Éxito: código listo. ✗ Fallo: requiere correcciones |

El archivo `.github/workflows/ci.yml` debe estar ubicado exactamente en esa ruta dentro del repositorio de GitHub. Una vez subido, GitHub Actions lo detecta y ejecuta las tareas automáticamente en la nube sin necesidad de configuración adicional.

---

## Dataset

- **Fuente**: [Telco Customer Churn — Kaggle (BlastChar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Filas**: ~7 043 clientes
- **Tasa de churn**: ~27%
- **Tarea**: Clasificación binaria (Churn: Yes/No)
