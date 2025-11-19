# Midterm Project: Bank Marketing Use case

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

## Data Set

1. Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).
2. Unzip the file.
3. Use the **bank-additional-full.csv** file (which contains all records and attributes).
4. Place the file in the project's **Data/** folder.

> **Note:** The dataset contains 41,188 records and 20 attributes

Data description:
| Variable | Type | Description |
| :--- | :--- | :--- |
| `age` | Numeric | Client's age. |
| `job` | Categorical | Type of job (e.g., "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"). |
| `marital` | Categorical | Marital status ("married", "divorced", "single"). Note: "divorced" includes divorced or widowed. |
| `education` | Categorical | Education level ("unknown", "secondary", "primary", "tertiary"). |
| `default` | Binary | Has credit in default? ("yes", "no"). |
| `balance` | Numeric | Average yearly balance, in euros. |
| `housing` | Binary | Has housing loan? ("yes", "no"). |
| `loan` | Binary | Has personal loan? ("yes", "no"). |
| `contact` | Categorical | Contact communication type ("unknown", "telephone", "cellular"). |
| `day` | Numeric | Last contact day of the month. |
| `month` | Categorical | Last contact month of year ("jan", "feb", ..., "dec"). |
| `duration` | Numeric | Last contact duration, in seconds. |
| `campaign` | Numeric | Number of contacts performed during this campaign and for this client (includes last contact). |
| `pdays` | Numeric | Number of days that passed by after the client was last contacted from a previous campaign (-1 means client was not previously contacted). |
| `previous` | Numeric | Number of contacts performed before this campaign and for this client. |
| `poutcome` | Categorical | Outcome of the previous marketing campaign ("unknown", "other", "failure", "success"). |
| **`y`** | **Binary** | **Target Variable: Has the client subscribed a term deposit? ("yes", "no").** |

## Project Structure

This repository is organized into two main components: exploration and production source code.

```text
dtc-ml-midterm-project/
│
├── conf/                       # Configuration files (Hydra)
│   └── config.yaml
│
├── data/                       # Raw and processed data
│   └── bank-additional-full.csv
│
├── notebooks/                  # Jupyter notebooks for experimentation
│   └── EDA.ipynb
│
├── src/                        # Source code for production
│   ├── main.py                 # FastAPI app entry point
│   ├── predict.py              # Client script for testing API
│   └── train.py                # Training pipeline & artifact generation
│
├── .env                        # Environment variables (Path configuration)
├── .gitignore                  # Files to ignore in Git
├── Dockerfile                  # Instructions to build the image
├── pyproject.toml              # Project dependencies (uv)
└── README.md                   # Project documentation
```

## Deployment & Usage

Run the following commands to build and start the FastAPI infrastructure container:

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