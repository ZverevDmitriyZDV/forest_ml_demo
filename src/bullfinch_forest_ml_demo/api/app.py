from __future__ import annotations

import os
from pathlib import Path
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


# ------------------------- env loading -------------------------

# 1) Позволяем выбрать env-файл через ENV_FILE
#    - Docker: можно НЕ задавать (compose подхватит env_file)
#    - Local: задаёшь ENV_FILE=.env.local
env_file = os.getenv("ENV_FILE")
if env_file and os.path.exists(env_file):
    load_dotenv(env_file, override=True)
else:
    load_dotenv(override=False)


# ------------------------- app -------------------------

app = FastAPI(title="Bullfinch Forest ML Demo API", version="0.1.0")
MODELS: Models | None = None


# ------------------------- path helpers -------------------------

def _project_root() -> Path:
    # .../src/bullfinch_forest_ml_demo/api/app.py -> parents[3] = project root
    return Path(__file__).resolve().parents[3]


def _in_docker() -> bool:
    # надёжные признаки контейнера
    if os.getenv("IN_DOCKER") == "1":
        return True
    if Path("/.dockerenv").exists():
        return True
    # иногда в контейнерах есть это
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists() and "docker" in cgroup.read_text(errors="ignore").lower():
            return True
    except Exception:
        pass
    return False


def _normalize_mlflow_uri(uri: str) -> str:
    """
    Поддерживаем один .env и для Docker, и для local:
    - В Docker: /app/mlruns существует -> оставляем как есть
    - Local (Windows/Mac/Linux): /app/mlruns НЕ существует -> подменяем на <project_root>/mlruns
    """
    if not uri:
        return uri

    # Нормализуем разные варианты записи
    # file:/app/mlruns, file:///app/mlruns, /app/mlruns...
    u = uri.strip()

    docker_prefixes = (
        "file:///app/mlruns",
        "file:/app/mlruns",
        "/app/mlruns",
    )

    if any(u.startswith(p) for p in docker_prefixes):
        if _in_docker():
            return u  # внутри контейнера это валидно

        # локально подменяем на реальный путь проекта
        local_mlruns = (_project_root() / "mlruns").resolve()
        local_base = local_mlruns.as_uri()  # file:///E:/.../mlruns или file:///... на unix

        # аккуратно сохраняем хвост пути после /app/mlruns
        tail = u
        for p in docker_prefixes:
            if tail.startswith(p):
                tail = tail[len(p):]
                break

        # tail может начинаться с /models/... или /7985.../models...
        # склеиваем: <local>/ + <tail without leading slash>
        tail = tail.lstrip("/\\")
        if tail:
            return f"{local_base}/{tail}".replace("\\", "/")
        return local_base

    # Если пользователь дал file:///<windows_path> или runs:/... — не трогаем
    return u


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


# ------------------------- feature/schema helpers -------------------------

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


def _get_expected_columns(model: Any) -> Optional[List[str]]:
    """
    Возвращает список колонок, которые ожидает pipeline/model на входе.
    Работает для sklearn>=1.0 через feature_names_in_.
    """
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    if hasattr(model, "named_steps"):
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
    return None


def _compute_derived(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Пытаемся вычислить недостающие engineered-features для trunk-forecasting.
    Работает даже если есть только часть lag-ов.
    """
    f = dict(features)

    if "trunk_deg" in f:
        td = float(f["trunk_deg"])

        if "trunk_deg_lag_1" in f and "trunk_deg_delta_1" not in f:
            f["trunk_deg_delta_1"] = td - float(f["trunk_deg_lag_1"])
        if "trunk_deg_lag_7" in f and "trunk_deg_delta_7" not in f:
            f["trunk_deg_delta_7"] = td - float(f["trunk_deg_lag_7"])

        def roll_stats(keys: List[str]) -> tuple[float, float]:
            vals = []
            for k in keys:
                if k in f and f[k] is not None:
                    vals.append(float(f[k]))
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

        if col.endswith("_l1"):
            base = col[:-3]
            if base in f:
                out[col] = f[base]
                continue

        out[col] = _default_value_for(col)

    df = pd.DataFrame([out], columns=expected_cols)

    if "planting_year" in df.columns:
        df["planting_year"] = pd.to_numeric(df["planting_year"], errors="coerce").fillna(0).astype(int)
    if "planting_year_l1" in df.columns:
        df["planting_year_l1"] = pd.to_numeric(df["planting_year_l1"], errors="coerce").fillna(0).astype(int)

    for c in df.columns:
        if c not in CAT_COLS_BASE and not c.endswith("_l1") and _is_numeric_col(c):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    for c in df.columns:
        if c in CAT_COLS_BASE or (c.endswith("_l1") and c[:-3] in CAT_COLS_BASE):
            df[c] = df[c].astype(str)

    return df


def _prepare_input_df(model: Any, features: Dict[str, Any]) -> pd.DataFrame:
    expected = _get_expected_columns(model)
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

    # Дефолты:
    # - Docker: file:///app/mlruns (volume)
    # - Local:  file:///<project_root>/mlruns
    default_tracking = "file:///app/mlruns" if _in_docker() else (_project_root() / "mlruns").as_uri()

    tracking_uri_raw = os.getenv("MLFLOW_TRACKING_URI", default_tracking)
    tracking_uri = _normalize_mlflow_uri(tracking_uri_raw)

    health_model_uri = _normalize_mlflow_uri(_require_env("HEALTH_MODEL_URI"))
    forecast_model_h1_uri = _normalize_mlflow_uri(_require_env("FORECAST_MODEL_H1_URI"))
    forecast_model_h7_uri_raw = os.getenv("FORECAST_MODEL_H7_URI")
    forecast_model_h7_uri = _normalize_mlflow_uri(forecast_model_h7_uri_raw) if forecast_model_h7_uri_raw else None

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
        X = _prepare_input_df(model, req.features)
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
        X = _prepare_input_df(model, req.features)
        y_pred = float(model.predict(X)[0])
    except ValueError as e:
        raise _schema_error_422(e, "/predict/trunk", model)

    return ForecastPredictResponse(horizon_days=req.horizon_days, y_pred=y_pred)
