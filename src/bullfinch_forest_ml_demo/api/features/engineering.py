from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import CAT_COLS_BASE, NUM_COLS_HINTS, DERIVED_FEATURES


def get_expected_columns(model: Any) -> Optional[List[str]]:
    """
    Returns a list of columns that the pipeline/model expects as input.
    For sklearn, feature_names_in_ is typically available.
    """
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))

    # if Pipeline — find step with feature_names_in_
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(getattr(step, "feature_names_in_"))

    return None


def is_numeric_col(col: str) -> bool:
    c = col.lower()
    return any(h in c for h in NUM_COLS_HINTS) and (col not in CAT_COLS_BASE) and not c.endswith("_l1")


def default_value_for(col: str) -> Any:
    if col.endswith("_l1"):
        base = col[:-3]
        if base in CAT_COLS_BASE:
            return "unknown"

    if col in CAT_COLS_BASE:
        return "unknown"

    if is_numeric_col(col) or any(x in col for x in ["_lag_", "_delta_", "_roll_", "_mean_", "_std_"]):
        return 0.0

    return None


def compute_derived(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    trying to calculate the missing engineered features for trunk-forecasting.
    """
    f = dict(features)

    if "trunk_deg" in f and f["trunk_deg"] is not None:
        td = float(f["trunk_deg"])

        if "trunk_deg_lag_1" in f and "trunk_deg_delta_1" not in f and f["trunk_deg_lag_1"] is not None:
            f["trunk_deg_delta_1"] = td - float(f["trunk_deg_lag_1"])

        if "trunk_deg_lag_7" in f and "trunk_deg_delta_7" not in f and f["trunk_deg_lag_7"] is not None:
            f["trunk_deg_delta_7"] = td - float(f["trunk_deg_lag_7"])

        def roll_stats(keys: List[str]) -> tuple[float, float]:
            vals = []
            for k in keys:
                if k in f and f[k] is not None:
                    vals.append(float(f[k]))
            vals.append(td)  # текущий trunk_deg
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


def align_to_expected(features: Dict[str, Any], expected_cols: List[str]) -> pd.DataFrame:
    """
    Creates a DataFrame with only the expected columns:
        - if *_l1 is expected, but base is present, we copy it
        - if derived trunk features are expected, we try to calculate it
        - everything else is filled with defaults
    """
    f = compute_derived(features)
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

        out[col] = default_value_for(col)

    df = pd.DataFrame([out], columns=expected_cols)

    # planting_year to int
    if "planting_year" in df.columns:
        df["planting_year"] = pd.to_numeric(df["planting_year"], errors="coerce").fillna(0).astype(int)
    if "planting_year_l1" in df.columns:
        df["planting_year_l1"] = pd.to_numeric(df["planting_year_l1"], errors="coerce").fillna(0).astype(int)

    # numeric cols -> float
    for c in df.columns:
        if c not in CAT_COLS_BASE and not c.endswith("_l1") and is_numeric_col(c):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    # categoricals -> str
    for c in df.columns:
        if c in CAT_COLS_BASE or (c.endswith("_l1") and c[:-3] in CAT_COLS_BASE):
            df[c] = df[c].astype(str)

    return df


def prepare_input_df(model: Any, features: Dict[str, Any]) -> pd.DataFrame:
    expected = get_expected_columns(model)
    if not expected:
        return pd.DataFrame([features])
    return align_to_expected(features, expected)
