## TO MANAGE DYNAMIC .ENV DATA

### STEP 1 - change [.configs/model_map.yaml](../configs/model_map.yaml)

    selection:
      health: lgbm 
      trunk_h1: sklearn
      trunk_h7: lgbm

### STEP 2 - generate .env.api

    python tools\select_models.py

### STEP 3 - generate k8s\.secret.env

    python tools\select_models.py --also-k8s-env k8s\secret.env

### STEP 4 - turn onn site local (don't forget to turn off before k8s)
    
    set ENV_FILE=.env.api
    python -m uvicorn bullfinch_forest_ml_demo.api.app:app --reload
