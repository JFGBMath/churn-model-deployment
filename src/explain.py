# src/explain.py

import os
import shap
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def get_top_shap_features(pipeline, input_df: pd.DataFrame, background_df: pd.DataFrame, n=5):
    """
    Computes SHAP values for a single prediction using the fitted pipeline's
    preprocessor + classifier, using a background sample for the explainer's
    reference distribution. Returns the top N features driving the prediction
    (by absolute SHAP value), with their direction of impact.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    background_transformed = preprocessor.transform(background_df)
    input_transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(classifier, background_transformed)
    shap_values = explainer.shap_values(input_transformed)[0]

    contributions = pd.Series(shap_values, index=feature_names)
    top_features = contributions.abs().sort_values(ascending=False).head(n)

    result = []
    for feature in top_features.index:
        result.append({
            "feature": feature,
            "impact": "increases" if contributions[feature] > 0 else "decreases",
            "shap_value": round(float(contributions[feature]), 4)
        })
    return result


def generate_explanation(top_features: list, prediction: str, probability: float, raw_input: dict) -> str:
    features_summary = "\n".join(
        f"- {f['feature']} {f['impact']} churn risk (SHAP value: {f['shap_value']})"
        for f in top_features
    )

    prompt = f"""A machine learning model predicted this customer's churn status as "{prediction}" with a probability of {probability:.1%}.

Customer's actual feature values:
{raw_input}

The top contributing factors (from SHAP analysis) were:
{features_summary}

Write a brief, plain-language explanation (2-3 sentences) of why the model made this prediction, suitable for a customer service representative with no data science background. Ground your explanation in the customer's actual feature values above — do not guess or assume values you weren't given. Do not mention SHAP or technical terms."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text