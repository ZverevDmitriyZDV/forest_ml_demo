from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from mlflow.tracking import MlflowClient

from training_task_classification import ClassificationConfig, run_task_classification
from training_task_forecasting import ForecastingConfig, run_task_forecasting

from mlflow_utils import (
    project_root_from_file,
    load_env,
    setup_mlflow_or_die,
    register_logged_model_and_set_alias,
    print_links,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _log_text_artifact(text: str, out_dir: Path, filename: str) -> None:
    _ensure_dir(out_dir)
    path = out_dir / filename
    path.write_text(text, encoding="utf-8")
    mlflow.log_artifact(str(path))


def _log_csv_artifact(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    _ensure_dir(out_dir)
    path = out_dir / filename
    df.to_csv(path, index=False)
    mlflow.log_artifact(str(path))


def _log_confusion_matrix_png(
    y_true,
    y_pred,
    labels: list[str],
    out_dir: Path,
    filename: str = "confusion_matrix.png",
) -> None:
    _ensure_dir(out_dir)
    path = out_dir / filename

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Task A (Health Classification)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    mlflow.log_artifact(str(path))


def main() -> None:
    project_root = project_root_from_file(__file__)
    load_env(project_root)

    print("ENV_FILE =", (None if not mlflow else None))  # чтобы не бесило: просто маркер
    print("ENV MLFLOW_TRACKING_URI =", mlflow.get_tracking_uri() if mlflow.get_tracking_uri() else None)

    client: MlflowClient = setup_mlflow_or_die()

    data_path = project_root / "data" / "processed" / "trees_merged.parquet"
    print("Using dataset:", data_path)
    print("MLflow tracking URI     =", mlflow.get_tracking_uri())

    mlflow.set_experiment("bullfinch_baselines")

    artifacts_root = project_root / "artifacts"
    _ensure_dir(artifacts_root)

    # ================= Task A: Classification =================
    cls_cfg = ClassificationConfig(data_path=data_path)

    with mlflow.start_run(run_name="task_a_classification") as run:
        cls_out: Dict[str, Any] = run_task_classification(cls_cfg)

        # params
        mlflow.log_param("task", "classification")
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("target", cls_cfg.target_col)
        mlflow.log_param("test_size", cls_cfg.test_size)
        mlflow.log_param("random_state", cls_cfg.random_state)

        # metrics
        mlflow.log_metric("macro_f1", float(cls_out["macro_f1"]))
        mlflow.log_metric("train_size", float(cls_out["n_train"]))
        mlflow.log_metric("test_size", float(cls_out["n_test"]))
        mlflow.log_metric("features_n", float(cls_out["features_n"]))

        print("\n=== Task A: Health Classification (LogReg baseline) ===")
        print("Macro F1:", cls_out["macro_f1"])
        print("Train/Test:", cls_out["n_train"], "/", cls_out["n_test"])
        print("Features:", cls_out["features_n"])

        # artifacts
        run_art_dir = artifacts_root / "task_a_classification"
        _log_text_artifact(str(cls_out["classification_report"]), run_art_dir, "classification_report.txt")

        y_test = cls_out.get("y_test")
        y_pred = cls_out.get("y_pred")
        X_test = cls_out.get("X_test")

        sample_n = 50
        if X_test is not None:
            df_sample = X_test.head(sample_n).copy()
            df_sample.insert(0, "y_true", list(y_test[:sample_n]))
            df_sample.insert(1, "y_pred", list(y_pred[:sample_n]))
        else:
            df_sample = pd.DataFrame({"y_true": list(y_test[:sample_n]), "y_pred": list(y_pred[:sample_n])})

        _log_csv_artifact(df_sample, run_art_dir, "predictions_sample.csv")

        labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
        _log_confusion_matrix_png(y_test, y_pred, labels, run_art_dir, "confusion_matrix.png")

        # ---- log model  ----
        # We take the logged-model URI directly - this removes the WARNING when register_model
        model_info = mlflow.sklearn.log_model(sk_model=cls_out["pipeline"], name="model")
        logged_uri = model_info.model_uri

        ver = register_logged_model_and_set_alias(
            client=client,
            logged_model_uri=logged_uri,
            registry_name="bullfinch-health",
            alias="prod",
        )
        print("Registered bullfinch-health version:", ver)
        print_links(run.info.run_id)

    # ================= Task B: Forecasting =================
    for h in (1, 7):
        fc_cfg = ForecastingConfig(data_path=data_path, horizon_days=h)

        with mlflow.start_run(run_name=f"task_b_forecasting_h{h}") as run:
            fc_out: Dict[str, Any] = run_task_forecasting(fc_cfg)

            mlflow.log_param("task", "forecasting")
            mlflow.log_param("model", "linear_regression")
            mlflow.log_param("target", fc_cfg.target_col)
            mlflow.log_param("horizon_days", int(fc_out["horizon_days"]))
            mlflow.log_param("lags", str(fc_cfg.lags))
            mlflow.log_param("rolling_windows", str(fc_cfg.rolling_windows))
            mlflow.log_param("test_fraction_by_time", float(fc_cfg.test_fraction_by_time))

            mlflow.log_metric("mae", float(fc_out["mae"]))
            mlflow.log_metric("rmse", float(fc_out["rmse"]))
            mlflow.log_metric("mae_naive", float(fc_out["mae_naive"]))
            mlflow.log_metric("rmse_naive", float(fc_out["rmse_naive"]))
            mlflow.log_metric("train_size", float(fc_out["n_train"]))
            mlflow.log_metric("test_size", float(fc_out["n_test"]))
            mlflow.log_metric("features_n", float(fc_out["features_n"]))

            print(f"\n=== Task B: Trunk Lean Forecasting (LinearRegression, h={h}) ===")
            print("MAE:", fc_out["mae"], " | RMSE:", fc_out["rmse"])

            # artifacts
            run_art_dir = artifacts_root / f"task_b_forecasting_h{h}"

            y_test = fc_out.get("y_test")
            y_pred = fc_out.get("y_pred")
            y_pred_naive = fc_out.get("y_pred_naive")
            test_df = fc_out.get("test_df")
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

            # log model + registry
            model_info = mlflow.sklearn.log_model(sk_model=fc_out["pipeline"], name="model")
            logged_uri = model_info.model_uri

            reg_name = f"bullfinch-health-h{h}"  # как у тебя
            ver = register_logged_model_and_set_alias(
                client=client,
                logged_model_uri=logged_uri,
                registry_name=reg_name,
                alias="prod",
            )
            print(f"Registered {reg_name} version:", ver)
            print_links(run.info.run_id)

    print("\nMLflow runs logged + models + artifacts saved successfully.")


if __name__ == "__main__":
    main()
