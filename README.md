# Midterm Project: Bank Marketing Use case

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

## Data Set

1. Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).
2. Unzip the file.
3. Use the **bank-additional-full.csv** file (which contains all records and attributes).
4. Place the file in the project's **Data/** folder.

> **Note:** The dataset contains 41,188 records and 20 attributes

Data description:
| Variable | Tipo | Descripción |
| :--- | :--- | :--- |
| `age` | Numérico | Edad del cliente. |
| `job` | Categórico | Tipo de trabajo (ej: "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"). |
| `marital` | Categórico | Estado civil ("married", "divorced", "single"). Nota: "divorced" incluye divorciados y viudos. |
| `education` | Categórico | Nivel educativo ("unknown", "secondary", "primary", "tertiary"). |
| `default` | Binario | ¿Tiene crédito en mora/incumplimiento? ("yes", "no"). |
| `balance` | Numérico | Saldo medio anual en euros. |
| `housing` | Binario | ¿Tiene préstamo hipotecario? ("yes", "no"). |
| `loan` | Binario | ¿Tiene préstamo personal? ("yes", "no"). |
| `contact` | Categórico | Tipo de comunicación de contacto ("unknown", "telephone", "cellular"). |
| `day` | Numérico | Último día del mes en que fue contactado. |
| `month` | Categórico | Último mes del año en que fue contactado ("jan", "feb", ..., "dec"). |
| `duration` | Numérico | Duración del último contacto en segundos. |
| `campaign` | Numérico | Número de contactos realizados durante esta campaña para este cliente (incluye el último contacto). |
| `pdays` | Numérico | Número de días que pasaron desde que el cliente fue contactado por última vez en una campaña anterior (-1 significa que no fue contactado previamente). |
| `previous` | Numérico | Número de contactos realizados antes de esta campaña para este cliente. |
| `poutcome` | Categórico | Resultado de la campaña de marketing anterior ("unknown", "other", "failure", "success"). |
| **`y`** | **Binario** | **Variable Objetivo (Target): ¿El cliente ha suscrito un depósito a plazo? ("yes", "no").** |

## Project Structure

This repository is organized into two main components: exploration and production source code.

├── notebooks/
│   └── EDA.ipynb     # Contains EDA, model selection, and metrics evaluation.
├── src/
│   ├── train.py      # Main training pipeline. Executes data cleaning, encoding, scaling, and model training. Generates the production artifact (.joblib).
│   └── predict.py    # Requests for model inference.
│   └── main.py    # FastAPI application entry point for model inference.
├── conf/             # Hydra configuration files.
└── Data/             # Raw dataset location.

## Deployment & Usage

Ejecutar el siguiente codigo para levantar la infraestructura de FastAPI con el modelo entrenado

```bash
docker build -t bank-marketing:latest .
docker run -p 8000:8000 bank-marketing:latest
```

The API will be available at [http://localhost:8000/docs](http://localhost:8000/docs)

Inference example:

```json
{
  "job": "admin.",
  "marital": "married",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "aug",
  "day_of_week": "thu",
  "age": 35,
  "duration": 210,
  "campaign": 1,
  "pdays": 999,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp_var_rate": 1.4,
  "cons_price_idx": 93.444,
  "cons_conf_idx": -36.1,
  "euribor3m": 4.963,
  "nr_employed": 5228.1
}
```