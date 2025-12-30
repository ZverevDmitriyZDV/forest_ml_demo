from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class HealthPredictRequest(BaseModel):
    # Любой набор фичей как key:value (и числовые, и категориальные)
    # Важно: ключи должны совпадать с колонками, на которых обучали пайплайн
    features: Dict[str, Any] = Field(..., description="Feature dict matching training columns")


class HealthPredictResponse(BaseModel):
    predicted_class: str
    probabilities: Optional[Dict[str, float]] = None


class ForecastPredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dict matching training columns")
    horizon_days: int = Field(1, description="Forecast horizon in days (e.g., 1 or 7)")


class ForecastPredictResponse(BaseModel):
    horizon_days: int
    y_pred: float
