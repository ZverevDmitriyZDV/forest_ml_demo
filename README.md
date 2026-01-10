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

## First-Time Setup and Run Guide
___

This repository contains an end-to-end ML system including:
- dataset generation and model training
- experiment tracking and model registry via MLflow
- inference API built with FastAPI
- local deployment using Docker and Kubernetes (Docker Desktop)

This document describes how to run the project **from scratch after cloning the repository**.


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

Change [setting file](./generation_dataset_tool/Bullfinch_Synthetic_Forestry_Dataset_Spec.xlsx)
```bash
python genaration_dataset_tool/generate_bullfinch_synthetic_forest_dataset.py
python src/bullfinch_forest_ml_demo/preprocessing/buid_datasets.py
```
Train models
```bash
python src/bullfinch_forest_ml_demo/training/train_baselines.py
python src/bullfinch_forest_ml_demo/training/train_baselines_lgbm.py
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

Manage models and change configuration [configs](./configs/model_map.yaml)
```
selection:
  health: lgbm
  trunk_h1: sklearn
  trunk_h7: lgbm
```

Generate ENV for Docker and K8s

```shell

    python tools\select_models.py

    python tools\select_models.py --also-k8s-env k8s\secret.env
    
```
Create Docker container if it's 1st start, or go next step
```
  docker build -t docker-api:latest -f docker/Dockerfile .
```

Apply Kubernetes manifests:

```bash
kubectl -n bullfinch apply -f k8s/
```
If the API was already deployed, restart it :

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
## 11. Common Issues
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

___

## HOW TO UPDATE - STEP BY STEP
___

### Main IDEA
- MLflow - has true models
- API - have no idea what is models, only have tag - http://bullfinch.local/docs

### HOW IT WORKS
- Learn new ML-model: get new metrics and artifacts
- Logging new model in Registry
- change settings and choose alias (prod, stagging)
- API get models in autopilot

### Why it's COOL to do so
- no need to change model,pdl by hands
- no need to change API
- no need to rebuild Docker-compose

### What exactly we're looking in MLflow
- experiments
- sheets runs (RMSE/MAE/MAPE/R2)
- compare runs
- artefacts (feature importance / plots / model.pkl)

We select the model with the lowest validation errors and stable behavior over the delay, and then label it with the alias prod.
    
### WE RESTART API ONLY!!!
- API read env file
    ```
    HEALTH_MODEL_URI=models:/bullfinch-health-lgbm@prod
    ```
- API asks MLflow what model is
- API download actual artifacts
- we use alias-based options
- we could change models without changing code
- we could upgrade logic and be sure everything works

### HOWEVER (current stage)
- API download models using alias @prod
- Alias use the latest version of models as approved version
- This means there is no automatic comparison, the latest version is not the best in real prod.
- "latest = the best metrics"  is accepted as an assumption
## UPDATE WORKFLOW
### STEP 0.1 - update model_map.yaml
run script for generate ENV files [(READ HOW)](./tools/README.md)

### STEP 1 - turn on DOCKER DESKTOP app
It's gonna save your day - promise.
CHECK
```
    docker info
    kubectl get nodes
```
Wanna see
```
    k8s - READY
```
### STEP 2 - MLflow START 
API download models after we start MLFLOW
```
    cd E:\Bullfintch_Earth\bullfinch-forest-ml-demo
    
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
CHECK 

    curl http:/127.0.0.1:5000
    
MLflow UI is working and ready as result

### STEP 3 - Model Registry in ready
```
    curl "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/search"
    curl "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/search"
```
Wanna see "model_version"

### STEP 4 - Start API in k8s    
If not started
```
    kubectl -n bullfinch apply -f k8s/
```
If started
```
    kubectl -n bullfinch rollout restart deploy/bullfinch-api
```
CHECKS
 ```   
    kubectl -n bullfinch get pods
    curl http://bullfinch.local/health
 ```
http://bullfinch.local/#docs

### Use prepared [examples](./src/bullfinch_forest_ml_demo/api/examples.md) for check POST METHODS



___
## Project Status
___
- Production-ready demo setup

- Model updates without code changes

- Suitable for technical and architectural presentations
