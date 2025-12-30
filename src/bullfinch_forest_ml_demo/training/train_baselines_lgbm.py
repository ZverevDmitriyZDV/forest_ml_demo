from __future__ import annotations

from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from training_task_classification_lgbm import (
    ClassificationLGBMConfig,
    run_task_classification_lgbm,
)
from training_task_forecasting_lgbm import (
    ForecastingLGBMConfig,
    run_task_forecasting_lgbm,
)

warnings.filterwarnings(
    "ignore",
    message="The filesystem tracking backend.*will be deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBM.* was fitted with feature names",
    category=UserWarning,
)


def _project_root() -> Path:
    # .../src/bullfinch_forest_ml_demo/training/train_baselines_lgbm.py
    return Path(__file__).resolve().parents[3]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _log_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
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
    project_root = _project_root()
    data_path = project_root / "data" / "processed" / "trees_merged.parquet"
    print("Using dataset:", data_path)

    # -------- MLflow tracking (Variant A: local ./mlruns) --------
    mlruns_dir = project_root / "mlruns"
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment("bullfinch_baselines")

    artifacts_root = project_root / "artifacts"
    _ensure_dir(artifacts_root)

    # ================= Task A: Classification (LightGBM) =================
    cls_cfg = ClassificationLGBMConfig(data_path=data_path)

    with mlflow.start_run(run_name="task_a_classification_lgbm"):
        out = run_task_classification_lgbm(cls_cfg)

        # Params
        mlflow.log_param("task", "classification")
        mlflow.log_param("model", "lightgbm_classifier")
        mlflow.log_param("target", "health_status")

        # Metrics
        mlflow.log_metric("macro_f1", out["macro_f1"])
        mlflow.log_metric("train_size", out["n_train"])
        mlflow.log_metric("test_size", out["n_test"])
        mlflow.log_metric("features_n", out["features_n"])

        # Model
        mlflow.sklearn.log_model(out["pipeline"], name="model")

        # ---- Artifacts: predictions sample (CSV) ----
        y_test = pd.Series(out["y_test"]).reset_index(drop=True)
        y_pred = pd.Series(out["y_pred"]).reset_index(drop=True)

        df_pred = pd.DataFrame({
            "y_true": y_test.astype(str),
            "y_pred": y_pred.astype(str),
        })

        # (опционально) добавим top-вероятности если есть predict_proba
        if "y_proba" in out and out["y_proba"] is not None and "labels" in out:
            proba = out["y_proba"]
            labels = out["labels"]
            proba_df = pd.DataFrame(proba, columns=[f"proba_{c}" for c in labels])
            df_pred = pd.concat([df_pred, proba_df], axis=1)

        # sample 200 строк чтобы файл был компактным
        df_pred_sample = df_pred.head(200)

        run_art_dir = artifacts_root / "task_a_classification_lgbm"
        _log_csv(df_pred_sample, run_art_dir, "predictions_sample.csv")

        # ---- Artifacts: confusion matrix PNG ----
        labels = out.get("labels") or sorted(pd.Series(out["y_test"]).unique().tolist())
        _log_confusion_matrix_png(out["y_test"], out["y_pred"], labels, run_art_dir)

        print("\n=== Task A: Health Classification (LightGBM) ===")
        print("Macro F1:", out["macro_f1"])

    # ================= Task B: Forecasting (LightGBM) =================
    for horizon in (1, 7):
        fc_cfg = ForecastingLGBMConfig(data_path=data_path, horizon_days=horizon)

        with mlflow.start_run(run_name=f"task_b_forecasting_lgbm_h{horizon}"):
            out = run_task_forecasting_lgbm(fc_cfg)

            mlflow.log_param("task", "forecasting")
            mlflow.log_param("model", "lightgbm_regressor")
            mlflow.log_param("target", "trunk_deg")
            mlflow.log_param("horizon_days", int(out["horizon_days"]))

            # metrics
            mlflow.log_metric("mae", out["mae"])
            mlflow.log_metric("rmse", out["rmse"])
            mlflow.log_metric("mae_naive", out["mae_naive"])
            mlflow.log_metric("rmse_naive", out["rmse_naive"])
            mlflow.log_metric("train_size", out["n_train"])
            mlflow.log_metric("test_size", out["n_test"])
            mlflow.log_metric("features_n", out["features_n"])

            mlflow.sklearn.log_model(out["pipeline"], name="model")

            # ---- Artifacts: predictions sample (CSV) ----
            y_test = pd.Series(out["y_test"]).reset_index(drop=True)
            y_pred = pd.Series(out["y_pred"]).reset_index(drop=True)
            y_pred_naive = pd.Series(out.get("y_pred_naive", [None] * len(y_test))).reset_index(drop=True)

            df_pred = pd.DataFrame({
                "y_true": y_test.astype(float),
                "y_pred": y_pred.astype(float),
                "y_pred_naive": pd.to_numeric(y_pred_naive, errors="coerce"),
                "abs_error": (y_test.astype(float) - y_pred.astype(float)).abs(),
            })

            if "test_df" in out and out["test_df"] is not None:
                meta = out["test_df"].reset_index(drop=True)
                df_pred = pd.concat([meta, df_pred], axis=1)

            run_art_dir = artifacts_root / f"task_b_forecasting_lgbm_h{horizon}"
            _log_csv(df_pred.head(200), run_art_dir, "predictions_sample.csv")

            print(f"\n=== Task B: Trunk Lean Forecasting (LightGBM, h={horizon}) ===")
            print("MAE:", out["mae"], "| RMSE:", out["rmse"])
            print("Naive MAE:", out["mae_naive"], "| Naive RMSE:", out["rmse_naive"])

    print("\nMLflow runs logged + models + artifacts saved successfully.")


if __name__ == "__main__":
    main()
