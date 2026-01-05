# HOW TO UPDATE - STEP BY STEP
___

## Main IDEA
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
___
# UPDATE WORKFLOW
___
## STEP 0.1 - update model_map.yaml
run script for generate ENV files [(READ HOW)](./tools/README.md)

## STEP 1 - turn on DOCKER DESKTOP app
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
## STEP 2 - MLflow START 
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

## STEP 3 - Model Registry in ready
```
    curl "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/search"
    curl "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/search"
```
Wanna see "model_version"

## STEP 4 - Start API in k8s    
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



