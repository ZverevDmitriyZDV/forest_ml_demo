# Bullfinch Forest ML Demo
## First-Time Setup and Run Guide

This repository contains an end-to-end ML system including:
- dataset generation and model training
- experiment tracking and model registry via MLflow
- inference API built with FastAPI
- local deployment using Docker and Kubernetes (Docker Desktop)

This document describes how to run the project **from scratch after cloning the repository**.

---

## 1. System Requirements

Make sure the following tools are installed and running:

### Required
- Windows 10 / 11
- Python 3.10+
- Git
- Docker Desktop
  - Kubernetes must be enabled in Docker Desktop settings
- kubectl (included with Docker Desktop)

###Verify installation:
```bash
docker info
kubectl get nodes
Expected result:
```
Docker is running

Kubernetes node is in Ready state

## 2. Clone Repository
```bash
git clone <REPOSITORY_URL>
cd bullfinch-forest-ml-demo
```

## 3. Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
````
Install dependencies:
```bash
pip install -e .
```
## 4. Project Structure Overview
```
bullfinch-forest-ml-demo/
│
├─ src/                        # Core source code (ML, API)
├─ notebooks/                  # Research notebooks
├─ mlflow_server/              # MLflow database and artifacts
│   ├─ mlflow.db
│   └─ artifacts/
├─ mlruns/                     # Local MLflow runs
├─ k8s/                        # Kubernetes manifests
├─ docker/                     # Docker-related files
├─ tools/                      # Utility scripts
└─ README.md
```

## 5. Start MLflow Server (MANDATORY)
MLflow must be running before the API, as the API loads models on startup.

Run from the project root:

```bash
venv\Scripts\python.exe -m mlflow server ^
  --host 0.0.0.0 ^
  --port 5000 ^
  --backend-store-uri sqlite:///mlflow_server/mlflow.db ^
  --registry-store-uri sqlite:///mlflow_server/mlflow.db ^
  --serve-artifacts ^
  --artifacts-destination file:///E:/Bullfintch_Earth/bullfinch-forest-ml-demo/mlflow_server/artifacts ^
  --allowed-hosts localhost,127.0.0.1,host.docker.internal,192.168.*,10.*,172.16.* ^
  --cors-allowed-origins "*"
```
Replace the path with the actual path to your project directory if needed.

### Verification
Open in browser:
```cpp
http://127.0.0.1:5000
```
If the MLflow UI loads, the server is running correctly.

## 6. Dataset Generation and Model Training (First Run)
Generate dataset
```bash
python tools/generate_dataset.py
```
Train models
```bash
python tools/train_models.py
```
After training:

- experiments are logged to MLflow

- models are registered in the MLflow Model Registry

- production aliases (e.g. prod) are assigned automatically

Verify models:

```bash
curl "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/search"
```
## 7. Reviewing Models and Metrics in MLflow
In the MLflow UI:

- Open Experiments

- Select the relevant experiment

- Compare metrics (RMSE, MAE, MAPE, R²)

- Ensure correct models have the prod alias

## 8. Deploy API to Kubernetes
Apply Kubernetes manifests:

```bash
kubectl -n bullfinch apply -f k8s/
```
If the API was already deployed, restart it:

```bash
kubectl -n bullfinch rollout restart deploy/bullfinch-api
```
## 9. API Health Checks
Pods status
```bash
kubectl -n bullfinch get pods
```
Expected:

```text
bullfinch-api-xxxx   1/1   Running
```
Health endpoint
```bash
curl http://bullfinch.local/health
```
Response:
```json
{"status":"ok"}
```
Swagger UI
Open in browser:

```arduino
http://bullfinch.local/docs
```
___
## 10. How Models Are Loaded by the API
- The API does NOT load local model files

- All models are loaded from MLflow Model Registry

Model URIs look like:

```text
models:/bullfinch-health-lgbm@prod
```
Meaning:

- bullfinch-health-lgbm — model name

- prod — alias pointing to the active version

Updating a model does NOT require API code changes — only alias updates and a deployment restart.

## 11. Updating Models After New Training [--FULL GUIDE--](./UPDATEML.md)
- Train a new model

- Register it in MLflow

- Assign alias prod

- Restart API:

___
## 12. Common Issues
### API crashes on startup
- MLflow is not running

- Model does not have prod alias

- Incorrect MLFLOW_TRACKING_URI

### MLflow not reachable from containers
Test connectivity:

```bash
kubectl -n bullfinch run netcheck --rm -it --image=curlimages/curl -- sh
curl http://host.docker.internal:5000
```
