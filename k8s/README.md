# Bullfinch Forest ML API — Kubernetes (Local)

## Требования

- Docker Desktop (Windows / Mac)
- Kubernetes включён в Docker Desktop
- kubectl установлен
- Python + venv (для локальной сборки образа)

Проверка:

    kubectl cluster-info

#1.Namespace:

    kubectl create namespace bullfinch

#2.Собрать Docker-образ локально:

    ⚠️ ВАЖНО: сборка должна идти в Docker Desktop context

    docker build -t docker-api:latest -f docker/Dockerfile .


###Проверка:

    docker images | findstr docker-api

#3.Secret с переменными окружения:

    Файл .env (пример):
    
        HEALTH_MODEL_URI=file:///app/mlruns/798548184845564305/models/m-xxx/artifacts
        FORECAST_MODEL_H1_URI=file:///app/mlruns/798548184845564305/models/m-yyy/artifacts
        FORECAST_MODEL_H7_URI=file:///app/mlruns/798548184845564305/models/m-zzz/artifacts


###Создать Secret:

    kubectl -n bullfinch create secret generic bullfinch-env --from-env-file=.env


###Проверка:

    kubectl -n bullfinch describe secret bullfinch-env

#4.Deployment + Service
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml


###Проверка:

    kubectl get pods -n bullfinch
    kubectl logs -n bullfinch deploy/bullfinch-api

#5.Ingress (NGINX)
Убедись, что ingress-nginx установлен:

        kubectl get pods -n ingress-nginx


###Применить ingress:

    kubectl apply -f k8s/ingress.yaml


###Проверка:

    kubectl get ingress -n bullfinch

#6.Прописать hosts (Windows)
    
    Открой Notepad от имени администратора
    Файл: C:\Windows\System32\drivers\etc\hosts
    Добавь строку: 127.0.0.1 bullfinch.local
    Сохрани файл.

#7.Проверка API:

    Swagger:    
    http://bullfinch.local/docs    
    
    Health:    
    curl http://bullfinch.local/health

##Диагностика:

###Pods

        kubectl get pods -n bullfinch
        kubectl describe pod -n bullfinch <pod-name>

###Logs

    kubectl logs -n bullfinch deploy/bullfinch-api
    kubectl logs -n bullfinch deploy/bullfinch-api --previous

###Exec внутрь контейнера

    kubectl exec -n bullfinch -it deploy/bullfinch-api -- sh

##Важно (архитектурные заметки)
    
hostPath используется ТОЛЬКО для local/dev
В production: S3 / MinIO initContainer + cache или MLflow Model Registry

# NONE HARDCODE MODE (Model Registry)

    python tools\select_models.py --also-k8s-env k8s\secret.env

    
####STEP1 - RUN MLFLOW correctly, kill previous
        mlflow server
            --host 0.0.0.0
            --port 5000
            --backend-store-uri "sqlite:///mlflow_server/mlflow.db"
            --registry-store-uri "sqlite:///mlflow_server/mlflow.db"
            --serve-artifacts
            --artifacts-destination "file:///E:/Bullfintch_Earth/bullfinch-forest-ml-demo/mlflow_server/artifacts"
            --allowed-hosts "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,host.docker.internal,host.docker.internal:5000,192.168.*,10.*,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,172.21.*,172.22.*,172.23.*,172.24.*,172.25.*,172.26.*,172.27.*,172.28.*,172.29.*,172.30.*,172.31.*"
            --cors-allowed-origins '*'
    
###STEP2 - k8s
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/secret-env.yaml

    kubectl -n bullfinch rollout restart deploy/bullfinch-api deployment.apps/bullfinch-api restarted
    kubectl -n bullfinch logs deploy/bullfinch-api --tail=200 -f
    kubectl -n bullfinch get pods -o wide


###STEP3 - CHECK URL
    http://bullfinch.local/docs#