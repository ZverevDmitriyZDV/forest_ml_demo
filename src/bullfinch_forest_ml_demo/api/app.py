from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Set

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import (
    HealthPredictRequest,
    HealthPredictResponse,
    ForecastPredictRequest,
    ForecastPredictResponse,
)
from .model_loader import load_models, Models


app = FastAPI(title="Bullfinch Forest ML Demo API", version="0.1.0")

MODELS: Models | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _features_to_df(features: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([features])


def _get_expected_input_columns(model: Any) -> Optional[Set[str]]:
    """
    Try to extract the exact set of input columns expected by the fitted pipeline.
    Works best for sklearn Pipeline with ColumnTransformer.
    """
    try:
        if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
            pre = model.named_steps["preprocess"]
            cols = getattr(pre, "feature_names_in_", None)
            if cols is not None:
                return set(map(str, cols))
    except Exception:
        pass
    return None


def _apply_l1_aliases(df: pd.DataFrame, expected: Set[str]) -> pd.DataFrame:
    """
    If training data used *_l1 column names (after merge), allow client to send without suffix.
    Example: species -> species_l1
    """
    for base in list(df.columns):
        l1 = f"{base}_l1"
        if l1 in expected and l1 not in df.columns:
            df[l1] = df[base]
    return df


def _align_to_expected_columns(df: pd.DataFrame, expected: Optional[Set[str]]) -> pd.DataFrame:
    """
    Make df safe for model.predict():
    - add missing columns as NaN (imputers handle)
    - drop extra columns not used by the model
    """
    if not expected:
        return df

    df = _apply_l1_aliases(df, expected)

    missing = expected - set(df.columns)
    for c in missing:
        df[c] = np.nan

    # keep only expected, in that order (stable)
    ordered = [c for c in df.columns if c in expected]
    # but ensure we include all expected
    ordered = list(expected) if len(ordered) != len(expected) else ordered
    # safer: deterministic alphabetical order
    ordered = sorted(list(expected))

    return df[ordered]


@app.on_event("startup")
def startup() -> None:
    global MODELS

    project_root = _project_root()
    tracking_uri = (project_root / "mlruns").as_uri()

    # >>> Replace with YOUR real run IDs <<<
    health_model_uri = "runs:/cd3fdb9680b14de190d5d30736814fd6/model"
    forecast_model_h1_uri = "runs:/71ccbd4218944c9ea51cb391ac984672/model"
    forecast_model_h7_uri = "runs:/2aa68418a42346f18955de3ce384d717/model"

    try:
        MODELS = load_models(
            tracking_uri=tracking_uri,
            health_model_uri=health_model_uri,
            forecast_model_h1_uri=forecast_model_h1_uri,
            forecast_model_h7_uri=forecast_model_h7_uri,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load models. "
            "Check MLflow run IDs and artifacts. "
            f"Original error: {e}"
        )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/health", response_model=HealthPredictResponse)
def predict_health(req: HealthPredictRequest) -> HealthPredictResponse:
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    model = MODELS.health_model
    X = _features_to_df(req.features)

    expected = _get_expected_input_columns(model)
    X = _align_to_expected_columns(X, expected)

    pred = model.predict(X)[0]

    probs_out = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = getattr(model.named_steps.get("model", None), "classes_", None)
        if classes is not None:
            probs_out = {str(c): float(p) for c, p in zip(classes, proba)}

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

    X = _features_to_df(req.features)
    expected = _get_expected_input_columns(model)
    X = _align_to_expected_columns(X, expected)

    y_pred = float(model.predict(X)[0])
    return ForecastPredictResponse(horizon_days=req.horizon_days, y_pred=y_pred)
