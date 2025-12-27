from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class ClassificationLGBMConfig:
    data_path: Path
    target_col: str = "health_status"

    test_size: float = 0.2
    random_state: int = 42

    id_cols: Tuple[str, ...] = ("tree_id",)
    drop_cols: Tuple[str, ...] = ("timestamp",)


def _load_data(cfg: ClassificationLGBMConfig) -> pd.DataFrame:
    df = pd.read_parquet(cfg.data_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}' in {cfg.data_path}")
    return df


def _select_features(df: pd.DataFrame, cfg: ClassificationLGBMConfig):
    drop = list(cfg.id_cols) + list(cfg.drop_cols)

    d = df.dropna(subset=[cfg.target_col]).copy()
    y = d[cfg.target_col].astype(str)

    X = d.drop(columns=[cfg.target_col], errors="ignore")
    X = X.drop(columns=[c for c in drop if c in X.columns], errors="ignore")

    # Prevent leakage from other targets
    for leak_col in ["estimated_age", "biomass", "risk_flag"]:
        if leak_col in X.columns:
            X = X.drop(columns=[leak_col], errors="ignore")

    return X, y


def _build_pipeline(X: pd.DataFrame, cfg: ClassificationLGBMConfig) -> Pipeline:
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    num_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_features),
            ("cat", cat_tf, categorical_features),
        ],
        remainder="drop",
    )

    # Good, safe defaults for demo
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=cfg.random_state,
        objective="multiclass",
        verbose=-1,
        n_jobs=1,
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def run_task_classification_lgbm(cfg: ClassificationLGBMConfig) -> Dict[str, object]:
    df = _load_data(cfg)
    X, y = _select_features(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pipe = _build_pipeline(X_train, cfg)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)

    labels = sorted(pd.Series(y_test).unique().tolist())

    return {
        "pipeline": pipe,  # или model
        "macro_f1": float(macro_f1),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_n": int(X_train.shape[1]),
        "y_test": y_test,  # array/Series
        "y_pred": y_pred,  # array
        "labels": labels,  # list[str]
    }

