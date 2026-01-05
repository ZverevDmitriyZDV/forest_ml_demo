# Bullfinch Forest ML API
___

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
___
## DOCKER + k8s + nginx - WHY?
- Docker
    - Packages the API with all dependencies
    - Works the same for everyone
- Kubernetes
    - Starts the container
    - Keeps it alive
    - Restarts on crash
    - Easy for scaling
- Nginx Ingress
    - Provides a nice URL (bullfinch.local)
    - Routes HTTP → pod
    - Easily add SSL
    
### System Architecture Overview
``` text
Client
    ↓
    Nginx Ingress
         ↓
        FastAPI (Kubernetes Pod)
             ↓
            MLflow Model Registry
                 ↓
                Model Artifacts Storage
```
___

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
___
## HOW TO START [(learn)](./INSTALL.md)
___
## HOW TO UPDATE ML [(learn)](./UPDATEML.md)
___
## Project Status
- Production-ready demo setup

- Model updates without code changes

- Suitable for technical and architectural presentations
___
