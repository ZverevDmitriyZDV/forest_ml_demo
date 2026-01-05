from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from .api_schemas import (
    HealthPredictRequest,
    HealthPredictResponse,
    ForecastPredictRequest,
    ForecastPredictResponse,
)
from .model_loader import load_models, Models
from .features.engineering import (
    get_expected_columns,
    prepare_input_df,
)

from .features.schema import build_schema_payload

# 1) ENV_FILE опционально
env_file = os.getenv("ENV_FILE")
if env_file and os.path.exists(env_file):
    load_dotenv(env_file, override=True)

app = FastAPI(title="Bullfinch Forest ML Demo API", version="0.1.0")
MODELS: Models | None = None


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _schema_error_422(e: Exception, endpoint: str, model: Any) -> HTTPException:
    expected = get_expected_columns(model) or []
    return HTTPException(
        status_code=422,
        detail={
            "error": "Input features do not match model schema",
            "endpoint": endpoint,
            "hint": "Add missing keys to req.features (or pass base keys; API will map base -> *_l1 and compute trunk deltas/rolling when possible).",
            "sklearn_error": str(e),
            "expected_columns_sample": expected[:40],
        },
    )


@app.on_event("startup")
def startup() -> None:
    global MODELS

    # ВАЖНО: не ломаем Docker.
    # - Docker: MLFLOW_TRACKING_URI=file:///app/mlruns
    # - Local: можно оставить пустым и дать file:///... на Windows
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")

    health_model_uri = _require_env("HEALTH_MODEL_URI")
    forecast_model_h1_uri = _require_env("FORECAST_MODEL_H1_URI")
    forecast_model_h7_uri = os.getenv("FORECAST_MODEL_H7_URI")  # optional

    MODELS = load_models(
        tracking_uri=tracking_uri,
        health_model_uri=health_model_uri,
        forecast_model_h1_uri=forecast_model_h1_uri,
        forecast_model_h7_uri=forecast_model_h7_uri,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> Dict[str, Any]:
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return build_schema_payload(
        health_model=MODELS.health_model,
        trunk_h1_model=MODELS.forecast_model_h1,
        trunk_h7_model=MODELS.forecast_model_h7,
    )



@app.post("/predict/health", response_model=HealthPredictResponse)
def predict_health(req: HealthPredictRequest) -> HealthPredictResponse:
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    model = MODELS.health_model

    try:
        X = prepare_input_df(model, req.features)
        pred = model.predict(X)[0]
    except ValueError as e:
        raise _schema_error_422(e, "/predict/health", model)

    probs_out = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                classes = getattr(model.named_steps.get("model", None), "classes_", None)
            if classes is not None:
                probs_out = {str(c): float(p) for c, p in zip(classes, proba)}
        except Exception:
            probs_out = None

    return HealthPredictResponse(predicted_class=str(pred), probabilities=probs_out)


@app.post("/predict/trunk", response_model=ForecastPredictResponse)
def predict_trunk(req: ForecastPredictRequest) -> ForecastPredictResponse:
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if req.horizon_days == 1:
        model = MODELS.forecast_model_h1
    elif req.horizon_days == 7:
        if MODELS.forecast_model_h7 is None:
            raise HTTPException(status_code=400, detail="Horizon 7 model is not configured")
        model = MODELS.forecast_model_h7
    else:
        raise HTTPException(status_code=400, detail="Supported horizon_days: 1 or 7")

    try:
        X = prepare_input_df(model, req.features)
        y_pred = float(model.predict(X)[0])
    except ValueError as e:
        raise _schema_error_422(e, "/predict/trunk", model)

    return ForecastPredictResponse(horizon_days=req.horizon_days, y_pred=y_pred)
