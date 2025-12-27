from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class ForecastingLGBMConfig:
    data_path: Path
    target_col: str = "trunk_deg"
    time_col: str = "timestamp"
    group_col: str = "tree_id"

    horizon_days: int = 1
    lags: Tuple[int, ...] = (1, 2, 7, 14)
    rolling_windows: Tuple[int, ...] = (7, 14)

    test_fraction_by_time: float = 0.2
    random_state: int = 42


def _load_data(cfg: ForecastingLGBMConfig) -> pd.DataFrame:
    df = pd.read_parquet(cfg.data_path)
    for col in [cfg.target_col, cfg.time_col, cfg.group_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {cfg.data_path}")
    df = df.copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    return df


def _make_lag_features(df: pd.DataFrame, cfg: ForecastingLGBMConfig) -> pd.DataFrame:
    d = df.sort_values([cfg.group_col, cfg.time_col]).copy()

    d["y_target"] = d.groupby(cfg.group_col)[cfg.target_col].shift(-cfg.horizon_days)

    for lag in cfg.lags:
        d[f"{cfg.target_col}_lag_{lag}"] = d.groupby(cfg.group_col)[cfg.target_col].shift(lag)

    for w in cfg.rolling_windows:
        d[f"{cfg.target_col}_roll_mean_{w}"] = (
            d.groupby(cfg.group_col)[cfg.target_col].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        )
        d[f"{cfg.target_col}_roll_std_{w}"] = (
            d.groupby(cfg.group_col)[cfg.target_col].shift(1).rolling(w).std().reset_index(level=0, drop=True)
        )

    d[f"{cfg.target_col}_delta_1"] = d.groupby(cfg.group_col)[cfg.target_col].diff(1)
    d[f"{cfg.target_col}_delta_7"] = d.groupby(cfg.group_col)[cfg.target_col].diff(7)

    d = d.dropna(subset=["y_target"]).copy()
    return d


def _time_split(df_feat: pd.DataFrame, cfg: ForecastingLGBMConfig):
    unique_dates = np.sort(df_feat[cfg.time_col].dt.date.unique())
    cutoff_idx = int(len(unique_dates) * (1.0 - cfg.test_fraction_by_time))
    cutoff_date = unique_dates[cutoff_idx]

    train = df_feat[df_feat[cfg.time_col].dt.date < cutoff_date].copy()
    test = df_feat[df_feat[cfg.time_col].dt.date >= cutoff_date].copy()
    return train, test


def _build_pipeline(X: pd.DataFrame, cfg: ForecastingLGBMConfig) -> Pipeline:
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    num_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_features),
            ("cat", cat_tf, categorical_features),
        ],
        remainder="drop",
    )

    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        objective="regression",
        random_state=cfg.random_state,
        verbose=-1,
        n_jobs=1,
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def _naive_persistence_baseline(test_df: pd.DataFrame, cfg: ForecastingLGBMConfig) -> np.ndarray:
    return test_df[cfg.target_col].astype(float).values


def run_task_forecasting_lgbm(cfg: ForecastingLGBMConfig) -> Dict[str, object]:
    df = _load_data(cfg)
    df_feat = _make_lag_features(df, cfg)

    # Keep rows where current target exists (for naive) and features exist
    df_feat = df_feat.dropna(subset=[cfg.target_col]).reset_index(drop=True)

    # Drop leakage cols: group id + future target
    drop_cols = [cfg.group_col, "y_target"]
    X_cols = [c for c in df_feat.columns if c not in drop_cols]

    train_df, test_df = _time_split(df_feat, cfg)

    X_train = train_df[X_cols].copy()
    y_train = train_df["y_target"].astype(float).values

    X_test = test_df[X_cols].copy()
    y_test = test_df["y_target"].astype(float).values

    # Naive baseline
    y_pred_naive = _naive_persistence_baseline(test_df, cfg)
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    rmse_naive = float(np.sqrt(mean_squared_error(y_test, y_pred_naive)))

    pipe = _build_pipeline(X_train, cfg)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "pipeline": pipe,
        "mae": float(mae),
        "rmse": float(rmse),
        "mae_naive": float(mae_naive),
        "rmse_naive": float(rmse_naive),
        "horizon_days": int(cfg.horizon_days),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features_n": int(X_train.shape[1]),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_naive": y_pred_naive,
        "test_df": test_df[[cfg.group_col, cfg.time_col, cfg.target_col]].copy(),  # удобно для CSV
    }

