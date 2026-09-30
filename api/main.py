# api/main.py

import joblib
import pandas as pd
from fastapi import FastAPI
from api.schemas import ChurnPredictionRequest, ChurnPredictionResponse, ExplainResponse
from src.monitoring import log_prediction

app = FastAPI(title="Churn Prediction API", version="1.0")

model = joblib.load("models/churn_model.pkl")
background_df = pd.read_csv("data/background_sample.csv")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest):
    input_dict = request.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    log_prediction(
        input_data=input_dict,
        prediction="Yes" if prediction == 1 else "No",
        probability=round(float(probability), 4)
    )

    return ChurnPredictionResponse(
        churn_prediction="Yes" if prediction == 1 else "No",
        churn_probability=round(float(probability), 4)
    )

from src.explain import get_top_shap_features, generate_explanation

@app.post("/explain", response_model=ExplainResponse)
def explain(request: ChurnPredictionRequest):
    input_dict = request.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    pred_label = "Yes" if prediction == 1 else "No"

    top_features = get_top_shap_features(model, input_df, background_df)
    explanation = explanation = generate_explanation(top_features, pred_label, probability, input_dict)

    return ExplainResponse(
        churn_prediction=pred_label,
        churn_probability=round(float(probability), 4),
        top_features=top_features,
        explanation=explanation
    )