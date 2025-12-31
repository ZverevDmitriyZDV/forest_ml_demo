from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from .schemas import (
    HealthPredictRequest,
    HealthPredictResponse,
    ForecastPredictRequest,
    ForecastPredictResponse,
)
from .model_loader import load_models, Models

# В docker-compose у тебя env_file уже подхватывает .env,
# но load_dotenv не мешает для локального запуска.
load_dotenv()

app = FastAPI(title="Bullfinch Forest ML Demo API", version="0.1.0")
MODELS: Models | None = None


# ------------------------- helpers -------------------------

CAT_COLS_BASE = {
    "species",
    "location_zone",
    "forest_type",
    "soil_type",
    "sensor_status",
}

NUM_COLS_HINTS = (
    "wind_exposure",
    "planting_year",
    "sap_flow_rate",
    "moisture_level",
    "temperature",
    "humidity",
    "leaf_color_index",
    "trunk_deg",
    "lag_",
    "delta_",
    "roll_",
    "_mean_",
    "_std_",
)


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _get_expected_columns(model: Any) -> Optional[List[str]]:
    """
    Возвращает список колонок, которые ожидает pipeline/model на входе.
    Работает для sklearn>=1.0 через feature_names_in_.
    """
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    # если это Pipeline
    if hasattr(model, "named_steps"):
        # попробуем найти step с feature_names_in_
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(getattr(step, "feature_names_in_"))
    return None


def _is_numeric_col(col: str) -> bool:
    c = col.lower()
    return any(h in c for h in NUM_COLS_HINTS) and (col not in CAT_COLS_BASE) and not c.endswith("_l1")


def _default_value_for(col: str) -> Any:
    if col.endswith("_l1"):
        base = col[:-3]
        if base in CAT_COLS_BASE:
            return "unknown"
    if col in CAT_COLS_BASE:
        return "unknown"
    if _is_numeric_col(col) or any(x in col for x in ["_lag_", "_delta_", "_roll_", "_mean_", "_std_"]):
        return 0.0
    # на всякий
    return None


def _compute_derived(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Пытаемся вычислить недостающие engineered-features для trunk-forecasting.
    Работает даже если есть только часть lag-ов.
    """
    f = dict(features)

    # delta
    if "trunk_deg" in f:
        td = float(f["trunk_deg"])
        if "trunk_deg_lag_1" in f and "trunk_deg_delta_1" not in f:
            f["trunk_deg_delta_1"] = td - float(f["trunk_deg_lag_1"])
        if "trunk_deg_lag_7" in f and "trunk_deg_delta_7" not in f:
            f["trunk_deg_delta_7"] = td - float(f["trunk_deg_lag_7"])

        # roll stats (приближение по доступным лагам)
        def roll_stats(keys: List[str]) -> tuple[float, float]:
            vals = []
            for k in keys:
                if k in f and f[k] is not None:
                    vals.append(float(f[k]))
            # добавим текущий trunk_deg если есть
            vals.append(td)
            arr = np.array(vals, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0))

        if "trunk_deg_roll_mean_7" not in f or "trunk_deg_roll_std_7" not in f:
            m, s = roll_stats(["trunk_deg_lag_7", "trunk_deg_lag_2", "trunk_deg_lag_1"])
            f.setdefault("trunk_deg_roll_mean_7", m)
            f.setdefault("trunk_deg_roll_std_7", s)

        if "trunk_deg_roll_mean_14" not in f or "trunk_deg_roll_std_14" not in f:
            m, s = roll_stats(["trunk_deg_lag_14", "trunk_deg_lag_7", "trunk_deg_lag_2", "trunk_deg_lag_1"])
            f.setdefault("trunk_deg_roll_mean_14", m)
            f.setdefault("trunk_deg_roll_std_14", s)

    return f


def _align_to_expected(features: Dict[str, Any], expected_cols: List[str]) -> pd.DataFrame:
    """
    Делает DataFrame строго с ожидаемыми колонками:
    - если ждут *_l1, а есть base -> копируем
    - если ждут derived trunk features -> пытаемся вычислить
    - всё остальное заполняем дефолтами
    """
    f = _compute_derived(features)

    out: Dict[str, Any] = {}

    for col in expected_cols:
        if col in f:
            out[col] = f[col]
            continue

        # base -> *_l1
        if col.endswith("_l1"):
            base = col[:-3]
            if base in f:
                out[col] = f[base]
                continue

        # если не нашли — дефолт
        out[col] = _default_value_for(col)

    df = pd.DataFrame([out], columns=expected_cols)

    # типовка: planting_year часто int
    if "planting_year" in df.columns:
        df["planting_year"] = pd.to_numeric(df["planting_year"], errors="coerce").fillna(0).astype(int)
    if "planting_year_l1" in df.columns:
        df["planting_year_l1"] = pd.to_numeric(df["planting_year_l1"], errors="coerce").fillna(0).astype(int)

    # numeric cols -> float
    for c in df.columns:
        if c not in CAT_COLS_BASE and not c.endswith("_l1") and _is_numeric_col(c):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    # categoricals -> str
    for c in df.columns:
        if c in CAT_COLS_BASE or (c.endswith("_l1") and c[:-3] in CAT_COLS_BASE):
            df[c] = df[c].astype(str)

    return df


def _prepare_input_df(model: Any, features: Dict[str, Any], endpoint: str) -> pd.DataFrame:
    expected = _get_expected_columns(model)

    # Если по какой-то причине модель не хранит schema, просто делаем df как есть
    if not expected:
        return pd.DataFrame([features])

    return _align_to_expected(features, expected)


def _schema_error_422(e: Exception, endpoint: str, model: Any) -> HTTPException:
    expected = _get_expected_columns(model) or []
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


# ------------------------- lifecycle -------------------------

@app.on_event("startup")
def startup() -> None:
    global MODELS

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")

    health_model_uri = _require_env("HEALTH_MODEL_URI")
    forecast_model_h1_uri = _require_env("FORECAST_MODEL_H1_URI")
    forecast_model_h7_uri = os.getenv("FORECAST_MODEL_H7_URI")  # optional

    MODELS = load_models(
        tracking_uri=tracking_uri,
        health_model_uri=health_model_uri,
        forecast_model_h1_uri=forecast_model_h1_uri,
        forecast_model_h7_uri=forecast_model_h7_uri,
    )


# ------------------------- endpoints -------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/health", response_model=HealthPredictResponse)
def predict_health(req: HealthPredictRequest) -> HealthPredictResponse:
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    model = MODELS.health_model

    try:
        X = _prepare_input_df(model, req.features, endpoint="/predict/health")
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
        X = _prepare_input_df(model, req.features, endpoint="/predict/trunk")
        y_pred = float(model.predict(X)[0])
    except ValueError as e:
        raise _schema_error_422(e, "/predict/trunk", model)

    return ForecastPredictResponse(horizon_days=req.horizon_days, y_pred=y_pred)
