from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# NOTE: matplotlib is used ONLY for saving confusion-matrix image as artifact
import matplotlib.pyplot as plt

from training_task_classification import (
    ClassificationConfig,
    run_task_classification,
)
from training_task_forecasting import (
    ForecastingConfig,
    run_task_forecasting,
)


def _project_root() -> Path:
    # .../src/bullfinch_forest_ml_demo/training/train_baselines.py
    return Path(__file__).resolve().parents[3]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _log_text_artifact(text: str, path: Path, filename: str) -> None:
    _ensure_dir(path)
    out = path / filename
    out.write_text(text, encoding="utf-8")
    mlflow.log_artifact(str(out))


def _log_csv_artifact(df: pd.DataFrame, path: Path, filename: str) -> None:
    _ensure_dir(path)
    out = path / filename
    df.to_csv(out, index=False)
    mlflow.log_artifact(str(out))


def _log_confusion_matrix_png(
    y_true,
    y_pred,
    labels: list[str],
    path: Path,
    filename: str = "confusion_matrix.png",
) -> None:
    _ensure_dir(path)
    out = path / filename

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Task A (Health Classification)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

    mlflow.log_artifact(str(out))


def main() -> None:
    project_root = _project_root()
    data_path = project_root / "data" / "processed" / "trees_merged.parquet"

    print("Using dataset:", data_path)

    # ---------------- MLflow setup ----------------
    # Variant A: filesystem backend in ./mlruns (fast for demo)
    mlruns_dir = project_root / "mlruns"
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment("bullfinch_baselines")

    artifacts_root = project_root / "artifacts"
    _ensure_dir(artifacts_root)

    # ================= Task A: Classification =================
    cls_cfg = ClassificationConfig(data_path=data_path)

    with mlflow.start_run(run_name="task_a_classification") as run:
        cls_out: Dict[str, Any] = run_task_classification(cls_cfg)

        # ---- params ----
        mlflow.log_param("task", "classification")
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("target", cls_cfg.target_col)
        mlflow.log_param("test_size", cls_cfg.test_size)
        mlflow.log_param("random_state", cls_cfg.random_state)

        # ---- metrics ----
        mlflow.log_metric("macro_f1", float(cls_out["macro_f1"]))
        mlflow.log_metric("train_size", float(cls_out["n_train"]))
        mlflow.log_metric("test_size", float(cls_out["n_test"]))
        mlflow.log_metric("features_n", float(cls_out["features_n"]))

        print("\n=== Task A: Health Classification (LogReg baseline) ===")
        print("Macro F1:", cls_out["macro_f1"])
        print("Train/Test:", cls_out["n_train"], "/", cls_out["n_test"])
        print("Features:", cls_out["features_n"])
        print("\nClassification report:\n", cls_out["classification_report"])

        # ---- artifacts ----
        run_art_dir = artifacts_root / "task_a_classification"
        _log_text_artifact(
            text=str(cls_out["classification_report"]),
            path=run_art_dir,
            filename="classification_report.txt",
        )

        # sample predictions (CSV)
        # Expect cls_out to return: y_test, y_pred, X_test (optional)
        y_test = cls_out.get("y_test")
        y_pred = cls_out.get("y_pred")
        X_test = cls_out.get("X_test")

        sample_n = 50
        if X_test is not None:
            df_sample = X_test.head(sample_n).copy()
            df_sample.insert(0, "y_true", list(y_test[:sample_n]))
            df_sample.insert(1, "y_pred", list(y_pred[:sample_n]))
        else:
            df_sample = pd.DataFrame(
                {"y_true": list(y_test[:sample_n]), "y_pred": list(y_pred[:sample_n])}
            )

        _log_csv_artifact(df_sample, run_art_dir, "predictions_sample.csv")

        # confusion matrix (PNG)
        labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
        _log_confusion_matrix_png(
            y_true=y_test,
            y_pred=y_pred,
            labels=labels,
            path=run_art_dir,
            filename="confusion_matrix.png",
        )

        # ---- log model ----
        # MLflow warning fix: use `name=` (artifact_path is deprecated)
        mlflow.sklearn.log_model(
            sk_model=cls_out["pipeline"],
            name="model",
        )

    # ================= Task B: Forecasting =================
    for h in (1, 7):
        fc_cfg = ForecastingConfig(data_path=data_path, horizon_days=h)
        with mlflow.start_run(run_name=f"task_b_forecasting_h{h}") as run:
            fc_out: Dict[str, Any] = run_task_forecasting(fc_cfg)

            # ---- params ----
            mlflow.log_param("task", "forecasting")
            mlflow.log_param("model", "linear_regression")
            mlflow.log_param("target", fc_cfg.target_col)
            mlflow.log_param("horizon_days", int(fc_out["horizon_days"]))
            mlflow.log_param("lags", str(fc_cfg.lags))
            mlflow.log_param("rolling_windows", str(fc_cfg.rolling_windows))
            mlflow.log_param("test_fraction_by_time", float(fc_cfg.test_fraction_by_time))

            # ---- metrics ----
            mlflow.log_metric("mae", float(fc_out["mae"]))
            mlflow.log_metric("rmse", float(fc_out["rmse"]))
            mlflow.log_metric("mae_naive", float(fc_out["mae_naive"]))
            mlflow.log_metric("rmse_naive", float(fc_out["rmse_naive"]))
            mlflow.log_metric("train_size", float(fc_out["n_train"]))
            mlflow.log_metric("test_size", float(fc_out["n_test"]))
            mlflow.log_metric("features_n", float(fc_out["features_n"]))

            print("\n=== Task B: Trunk Lean Forecasting (LinearRegression lag baseline) ===")
            print("MAE:", fc_out["mae"], " | RMSE:", fc_out["rmse"])
            print("Naive MAE:", fc_out["mae_naive"], " | Naive RMSE:", fc_out["rmse_naive"])
            print("Train/Test:", fc_out["n_train"], "/", fc_out["n_test"])
            print("Features:", fc_out["features_n"])
            print("Horizon days:", fc_out["horizon_days"])

            # ---- artifacts ----
            run_art_dir = artifacts_root / f"task_b_forecasting_h{h}"

            # sample predictions (CSV)
            # Expect fc_out to return: y_test, y_pred, y_pred_naive, X_test(optional), test_df(optional)
            y_test = fc_out.get("y_test")
            y_pred = fc_out.get("y_pred")
            y_pred_naive = fc_out.get("y_pred_naive")

            # If you return a test dataframe with ids/timestamps, we’ll include them.
            test_df = fc_out.get("test_df")  # optional: dataframe with timestamp/tree_id/target
            sample_n = 100

            if isinstance(test_df, pd.DataFrame):
                df_sample = test_df.head(sample_n).copy()
                df_sample["y_true"] = list(y_test[:sample_n])
                df_sample["y_pred"] = list(y_pred[:sample_n])
                df_sample["y_pred_naive"] = list(y_pred_naive[:sample_n])
            else:
                df_sample = pd.DataFrame(
                    {
                        "y_true": list(y_test[:sample_n]),
                        "y_pred": list(y_pred[:sample_n]),
                        "y_pred_naive": list(y_pred_naive[:sample_n]),
                    }
                )

            _log_csv_artifact(df_sample, run_art_dir, "predictions_sample.csv")

            # ---- log model ----
            mlflow.sklearn.log_model(
                sk_model=fc_out["pipeline"],
                name="model",
            )

    print("\nMLflow runs logged + models + artifacts saved successfully.")


if __name__ == "__main__":
    main()
