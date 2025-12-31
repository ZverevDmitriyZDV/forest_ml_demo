python -m uvicorn bullfinch_forest_ml_demo.api.app:app --reload
docker compose -f docker/docker-compose.yml logs -f api
docker compose -f docker/docker-compose.yml logs -f api

http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs

check healthy
{
  "features": {
    "species": "Douglas-fir",
    "location_zone": "coastal",
    "forest_type": "coniferous",
    "soil_type": "loam",
    "planting_year": 2012,
    "trunk_deg": 2.1,
    "sap_flow_rate": 1.15,
    "moisture_level": 0.42,
    "temperature": 15.8,
    "humidity": 0.76,
    "leaf_color_index": 0.86,
    "wind_exposure": 0.45,
    "sensor_status": "ok"
  }
}



check unhealthy
{
  "features": {
    "species": "Pine",
    "location_zone": "interior",
    "forest_type": "mixed",
    "soil_type": "sandy",
    "planting_year": 2005,
    "wind_exposure": 0.85,
    "trunk_deg": 7.8,
    "sap_flow_rate": 0.35,
    "moisture_level": 0.12,
    "temperature": 29.5,
    "humidity": 0.28,
    "leaf_color_index": 0.41,  
    "sensor_status": "ok"
    
  }
}


POST /predict/trunk (h=1 and h=7) change days "horizon_days": 7
{
  "horizon_days": 1,
  "features": {
    "species": "Douglas-fir",
    "location_zone": "coastal",
    "forest_type": "coniferous",
    "soil_type": "loam",
    "planting_year": 2012,
    "wind_exposure": 0.6,
    "sensor_status": "ok",
    "trunk_deg": 2.1,
    "trunk_deg_lag_1": 2.0,
    "trunk_deg_lag_2": 2.2,
    "trunk_deg_lag_7": 2.5,
    "trunk_deg_lag_14": 2.7,
    "sap_flow_rate": 1.10,
    "moisture_level": 0.44,
    "temperature": 16.2,
    "humidity": 0.74,
    "leaf_color_index": 0.85
  }
}

start docker
docker compose -f docker/docker-compose.yml up --build

stop docker
docker compose -f docker/docker-compose.yml down

logs
docker compose -f docker/docker-compose.yml logs -f api

