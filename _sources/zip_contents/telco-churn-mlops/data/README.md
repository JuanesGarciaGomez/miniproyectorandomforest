# Carpeta `data/`

Esta carpeta debe contener el archivo `telco_churn.csv` con el dataset
**Telco Customer Churn** de Kaggle.

## Cómo obtenerlo

### Opción 1 — Descarga manual
1. Ir a https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Descargar el archivo `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Renombrarlo a `telco_churn.csv` y dejarlo dentro de esta carpeta `data/`.

### Opción 2 — Kaggle CLI
```bash
pip install kaggle
# coloca tu kaggle.json en ~/.kaggle/
kaggle datasets download -d blastchar/telco-customer-churn -p data/ --unzip
mv data/WA_Fn-UseC_-Telco-Customer-Churn.csv data/telco_churn.csv
```

## Esquema esperado

- **7043 filas, 21 columnas** aproximadamente.
- Variable objetivo: `Churn` (`Yes` / `No`).
- Mezcla de variables categóricas (ej. `Contract`, `PaymentMethod`,
  `InternetService`) y numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`).
- `TotalCharges` viene como string y contiene espacios en blanco que hay
  que convertir a NaN — eso se hace en `1_eda_preprocessing.ipynb`.

> **Nota sobre datos**: este repositorio no incluye el CSV para respetar
> los términos de uso de Kaggle. El `.gitignore` excluye `data/*.csv`.
