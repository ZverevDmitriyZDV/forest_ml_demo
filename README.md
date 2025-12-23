## Tech Stack Overview

- Python — core language for ML and backend.
- PostgreSQL — example production-grade storage.
- MLflow — experiment tracking and model artifacts.
- FastAPI — inference service.
- Docker — containerized environments.
- Kubernetes (K8s) — orchestration and scaling.
- NGINX — deployment / ingress layer.

### Note on Vector Databases
Vector DBs are not used in this demo because tasks are tabular.
They become relevant when adding embeddings (images/text) or similarity search.

## Implemented ML Tasks

### Task A — Health Classification
- Models: Logistic Regression, LightGBM
- Metric: Macro F1

### Task B — Trunk Lean Forecasting
- Models: Naive baseline, LightGBM Regressor
- Metrics: MAE, RMSE

## MLOps
- All experiments and models are tracked in MLflow.
- Dataset regeneration enables retraining demos.
