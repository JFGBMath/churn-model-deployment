# api/main.py

import joblib
import pandas as pd
from fastapi import FastAPI
from api.schemas import ChurnPredictionRequest, ChurnPredictionResponse

app = FastAPI(title="Churn Prediction API", version="1.0")

model = joblib.load("models/churn_model.pkl")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest):
    input_dict = request.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return ChurnPredictionResponse(
        churn_prediction="Yes" if prediction == 1 else "No",
        churn_probability=round(float(probability), 4)
    )