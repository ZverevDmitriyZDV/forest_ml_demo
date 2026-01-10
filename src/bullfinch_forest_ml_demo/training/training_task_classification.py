from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ClassificationConfig:
    data_path: Path
    target_col: str = "health_status"

    # We do time-aware split in forecasting; for classification we do a stratified split.
    test_size: float = 0.2
    random_state: int = 42

    # Useful columns (adjust if needed)
    id_cols: Tuple[str, ...] = ("tree_id",)
    drop_cols: Tuple[str, ...] = ("timestamp",)  # timestamp usually not needed for this baseline


def _load_data(cfg: ClassificationConfig) -> pd.DataFrame:
    df = pd.read_parquet(cfg.data_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}' in {cfg.data_path}")
    return df


def _select_features(df: pd.DataFrame, cfg: ClassificationConfig) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Keep only rows with target
    d = df.dropna(subset=[cfg.target_col]).copy()

    y = d[cfg.target_col].astype(str)

    # store IDs for later artifacts
    tree_id = d["tree_id"].copy() if "tree_id" in d.columns else pd.Series([None] * len(d))

    X = d.drop(columns=[cfg.target_col], errors="ignore")

    # drop cols that shouldn't be features
    drop = list(cfg.id_cols) + list(cfg.drop_cols)
    X = X.drop(columns=[c for c in drop if c in X.columns], errors="ignore")

    # drop other targets to avoid leakage
    for leak_col in ["estimated_age", "biomass", "risk_flag"]:
        if leak_col in X.columns:
            X = X.drop(columns=[leak_col], errors="ignore")

    return X, y, tree_id


def _build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Split columns by dtype
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),  # safe for sparse
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    return pipe


def run_task_classification(cfg: ClassificationConfig) -> Dict[str, object]:
    df = _load_data(cfg)
    X, y , tree_id = _select_features(df, cfg)

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        tree_id,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pipe = _build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    y_proba = None
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)

    labels = sorted(y.unique().tolist())  # stable order for metrics/plots
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)

    return {
        "model_name": "logreg_baseline",
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "pipeline": pipe,

        # for artifacts
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "labels": labels,
        "confusion_matrix": cm,

        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_n": int(X.shape[1]),
        "classes": labels,
    }
