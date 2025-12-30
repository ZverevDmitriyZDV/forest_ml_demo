from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlflow
import mlflow.sklearn


@dataclass
class Models:
    health_model: object
    forecast_model_h1: object
    forecast_model_h7: Optional[object] = None


def load_models(
    tracking_uri: str,
    health_model_uri: str,
    forecast_model_h1_uri: str,
    forecast_model_h7_uri: Optional[str] = None,
) -> Models:
    """
    Loads MLflow-logged sklearn-compatible models (pipelines).
    """
    mlflow.set_tracking_uri(tracking_uri)

    health_model = mlflow.sklearn.load_model(health_model_uri)
    forecast_model_h1 = mlflow.sklearn.load_model(forecast_model_h1_uri)

    forecast_model_h7 = None
    if forecast_model_h7_uri:
        forecast_model_h7 = mlflow.sklearn.load_model(forecast_model_h7_uri)

    return Models(
        health_model=health_model,
        forecast_model_h1=forecast_model_h1,
        forecast_model_h7=forecast_model_h7,
    )
