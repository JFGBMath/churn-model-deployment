# Churn Model Deployment

End-to-end machine learning project: from model training to a production-ready,
containerized API with automated testing and monitoring. Built to demonstrate the
full lifecycle of taking a data science solution from experimentation to
production,testing it and preparing it for deployment.

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1. Model Training | EDA + churn classification model | ✅ Done |
| 2. API | FastAPI service serving predictions | ✅ Done |
| 3. Containerization | Docker + docker-compose | 🚧 In progress |
| 4. Testing & CI | pytest + GitHub Actions | ⬜ Pending |
| 5. Monitoring | Prediction logging + data drift detection | ⬜ Pending |
| 6. LLM Explainability | Natural-language prediction explanations | ⬜ Pending |
| 7. Kubernetes | Local deployment manifests | ⬜ Pending |

## Problem

Predicting customer churn for a telecom provider, using the IBM-extended
version of the Telco Customer Churn dataset (7,043 customers, 33 raw columns).
The goal is to flag customers likely to cancel their service so retention
efforts can be targeted a classic, high-value business use case for
classification models.

Columns that leak post-outcome information (`Churn Score`, `CLTV`,
`Churn Reason`, and the duplicate `Churn Label`) were identified and removed
before modeling — these are either derived from another model's predictions
or only populated for customers who already churned, and would make any
model trained on them look artificially strong while being useless in
production, where that information isn't available at prediction time.

## Approach

- **Baseline overall churn rate**: 26.54% (moderate class imbalance 
  `class_weight="balanced"` used in both candidate models).
- **Models compared**: Logistic Regression vs. Random Forest.
- **Selection metric**: recall on the churn class, not accuracy in this
  business context, a missed churner (false negative) is more costly than
  a false alarm (unnecessary retention outreach).

| Model | Churn Recall | Churn Precision | ROC-AUC |
|---|---|---|---|
| **Logistic Regression** (chosen) | **0.78** | 0.51 | **0.8489** |
| Random Forest | 0.52 | 0.63 | 0.8322 |

Logistic Regression was selected for catching significantly more actual
churners, despite a higher false-positive rate — the right trade-off for
this use case.

## Tech Stack

- **Model**: scikit-learn (Logistic Regression, `ColumnTransformer` pipeline)
- **API**: FastAPI + Pydantic v2
- **Testing**: pytest, FastAPI `TestClient`
- **Containerization**: Docker, docker-compose *(in progress)*
- **CI/CD**: GitHub Actions *(planned)*
- **Monitoring**: Evidently (data drift), custom prediction logging *(planned)*
- **Explainability**: SHAP + LLM via Anthropic API *(planned)*
- **Orchestration**: Kubernetes, local via Minikube *(planned)*

## Repository Structure

churn-model-deployment/
├── api/ # FastAPI service
│ ├── main.py # App, /health and /predict endpoints
│ └── schemas.py # Pydantic request/response models
├── data/raw/ # Raw dataset (IBM Telco Customer Churn, extended)
├── models/ # Serialized trained model (churn_model.pkl)
├── notebooks/ # EDA and model training notebook
├── src/
│ └── data_pipeline.py # load_data, clean_data, feature/preprocessing helpers
├── tests/
│ └── test_api.py # API endpoint tests
├── k8s/ # Kubernetes manifests (Phase 7)
└── .github/workflows/ # CI pipeline (Phase 4)


## API

### `GET /health`
Returns `{"status": "ok"}` — basic liveness check.

### `POST /predict`
Accepts customer feature data, returns a churn prediction and probability.

**Example request:**
```json
{
  "Gender": "Male",
  "Senior Citizen": "No",
  "Partner": "No",
  "Dependents": "No",
  "Tenure Months": 100,
  "Phone Service": "Yes",
  "Multiple Lines": "No",
  "Internet Service": "DSL",
  "Online Security": "Yes",
  "Online Backup": "Yes",
  "Device Protection": "No",
  "Tech Support": "No",
  "Streaming TV": "No",
  "Streaming Movies": "No",
  "Contract": "Month-to-month",
  "Paperless Billing": "Yes",
  "Payment Method": "Mailed check",
  "Monthly Charges": 0,
  "Total Charges": 0
}
```

**Example response:**
```json
{
  "churn_prediction": "No",
  "churn_probability": 0.028
}
```

All fields are validated with Pydantic (`Literal` types for categoricals,
numeric bounds for continuous fields) invalid input returns a `422` error
instead of silently reaching the model.

## How to Reproduce

```bash
git clone https://github.com/JFGBMath/churn-model-deployment.git
cd churn-model-deployment

python -m venv venv
source venv/Scripts/activate   # Windows Git Bash; use venv\Scripts\activate on CMD

pip install pandas numpy scikit-learn jupyter matplotlib seaborn
pip install fastapi uvicorn pydantic httpx pytest

# Train the model (or use the one already in models/)
jupyter notebook notebooks/01_eda_and_model_training.ipynb

# Run the API
uvicorn api.main:app --reload
# Visit http://127.0.0.1:8000/docs for interactive Swagger UI

# Run tests
pytest tests/ -v
```

## License

MIT



## Author

Jesús Fernando Gómez Brito ([Linkedin](www.linkedin.com/in/jesús-fernando-gómez-brito-02a895279))