import joblib
import pandas as pd
from fastapi import FastAPI
from typing import Annotated
from pydantic import BaseModel, StringConstraints, Field
import uvicorn
try:
    from src.train import DataPreprocessor
except ImportError:
    from train import DataPreprocessor

with open('model_production.joblib', 'rb') as model_file:
        model = joblib.load(model_file)

class PredictRequest(BaseModel):
    job: Annotated[str, Field(min_length=1, max_length=20)]
    marital: Annotated[str, Field(min_length=1, max_length=20)]
    education: Annotated[str, Field(min_length=1, max_length=30)]
    default: Annotated[str, Field(min_length=1, max_length=10)]
    housing: Annotated[str, Field(min_length=1, max_length=10)]
    loan: Annotated[str, Field(min_length=1, max_length=10)]
    contact: Annotated[str, Field(min_length=1, max_length=20)]
    month: Annotated[str, Field(min_length=1, max_length=10)]
    day_of_week: Annotated[str, Field(min_length=1, max_length=3)]
    age: int
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: Annotated[str, Field(min_length=1, max_length=20)]
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

class PredictionRespons(BaseModel):
    predict: int
    label: Annotated[str, Field(min_length=1, max_length=3)]

app = FastAPI(title="record-prediction-api")

@app.post("/predict", response_model=PredictionRespons)
def predict(record: PredictRequest):

    data_dict = record.model_dump()

    input_data = {
        "age": data_dict["age"],
        "job": data_dict["job"],
        "marital": data_dict["marital"],
        "education": data_dict["education"],
        "default": data_dict["default"],
        "housing": data_dict["housing"],
        "loan": data_dict["loan"],
        "contact": data_dict["contact"],
        "month": data_dict["month"],
        "day_of_week": data_dict["day_of_week"],
        "duration": data_dict["duration"],
        "campaign": data_dict["campaign"],
        "pdays": data_dict["pdays"],
        "previous": data_dict["previous"],
        "poutcome": data_dict["poutcome"],
        "emp.var.rate": data_dict["emp_var_rate"],
        "cons.price.idx": data_dict["cons_price_idx"],
        "cons.conf.idx": data_dict["cons_conf_idx"],
        "euribor3m": data_dict["euribor3m"],
        "nr.employed": data_dict["nr_employed"]
    }

    df = pd.DataFrame([input_data])
    predict = model.predict(df)[0]

    return {
            'predict': int(predict),
            'label': 'yes' if predict == 1 else 'no'
        }

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
