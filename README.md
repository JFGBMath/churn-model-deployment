# Churn Model Deployment

End-to-end machine learning project: from model training to a production-ready,
containerized API with monitoring and CI/CD. Built to demonstrate the full
lifecycle of taking a data science solution from experimentation to production.

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1. Model Training | EDA + churn classification model | 🚧 In progress |
| 2. API | FastAPI service serving predictions | ⬜ Pending |
| 3. Containerization | Docker + docker-compose | ⬜ Pending |
| 4. Testing & CI | pytest + GitHub Actions | ⬜ Pending |
| 5. Monitoring | Prediction logging + data drift detection | ⬜ Pending |
| 6. LLM Explainability | Natural-language prediction explanations | ⬜ Pending |
| 7. Kubernetes | Local deployment manifests | ⬜ Pending |

## Problem



## Tech Stack

- **Model**: scikit-learn / XGBoost
- **API**: FastAPI + Pydantic
- **Containerization**: Docker, docker-compose
- **CI/CD**: GitHub Actions
- **Monitoring**: Evidently (data drift), custom prediction logging
- **Explainability**: SHAP + LLM (Anthropic API)
- **Orchestration**: Kubernetes (local, via Minikube)

## Repository Structure

churn-model-deployment/
├── data/raw/ # Raw dataset
├── notebooks/ # EDA and model training notebook
├── src/ # Reusable ML logic (data pipeline, training, monitoring, explain)
├── api/ # FastAPI service
├── models/ # Serialized trained model
├── tests/ # Unit and API tests
├── k8s/ # Kubernetes manifests
└── .github/workflows/ # CI pipeline


## How to Reproduce


```bash
git clone https://github.com/JFGBMath/churn-model-deployment.git
cd churn-model-deployment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## License

MIT