# src/bullfinch_forest_ml_demo/api/features/schema.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constants import CAT_COLS_BASE,DERIVED_FEATURES
from .examples import PREDICT_HEALTH,TRUNK_EXAMPLE
from .engineering import get_expected_columns


def build_schema_payload(
    *,
    health_model: Any,
    trunk_h1_model: Any,
    trunk_h7_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Собирает стабильный JSON-контракт для /schema:
    - base categorical keys
    - derived features which API can compute
    - expected columns for each loaded model (из sklearn feature_names_in_)
    - примеры запросов
    """

    health_cols: List[str] = get_expected_columns(health_model) or []
    trunk_h1_cols: List[str] = get_expected_columns(trunk_h1_model) or []
    trunk_h7_cols: List[str] = []
    if trunk_h7_model is not None:
        trunk_h7_cols = get_expected_columns(trunk_h7_model) or []

    return {
        "categorical_base": sorted(list(CAT_COLS_BASE)),
        "base_to_l1_mapping_rule": (
            "If model expects '<col>_l1' and request provides '<col>', API copies it."
        ),
        "derived_features_auto": sorted(list(DERIVED_FEATURES)),
        "models": {
            "health": {"expected_columns": health_cols},
            "trunk_h1": {"expected_columns": trunk_h1_cols},
            "trunk_h7": {"expected_columns": trunk_h7_cols},
        },
        "examples": {
            "predict_health": PREDICT_HEALTH,
            "predict_trunk": TRUNK_EXAMPLE,
        },
    }
