import requests

url = 'http://localhost:8000/predict'

client = {
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

response = requests.post(url, json=client).json()

print(f"predict: {response['predict']}")