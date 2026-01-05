@echo off
setlocal
cd /d E:\Bullfintch_Earth\bullfinch-forest-ml-demo

mlflow server ^
  --host 0.0.0.0 ^
  --port 5000 ^
  --backend-store-uri "sqlite:///mlflow_server/mlflow.db" ^
  --default-artifact-root "file:///E:/Bullfintch_Earth/bullfinch-forest-ml-demo/mlflow_server/artifacts" ^
  --allowed-hosts "localhost,127.0.0.1,host.docker.internal,127.0.0.1:5000,host.docker.internal:5000,192.168.*,10.*,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,172.21.*,172.22.*,172.23.*,172.24.*,172.25.*,172.26.*,172.27.*,172.28.*,172.29.*,172.30.*,172.31.*" ^
  --cors-allowed-origins "*"