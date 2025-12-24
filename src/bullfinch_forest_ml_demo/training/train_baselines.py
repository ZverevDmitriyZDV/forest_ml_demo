from __future__ import annotations

from pathlib import Path

from training_task_classification import (
    ClassificationConfig,
    run_task_classification,
)
from training_task_forecasting import (
    ForecastingConfig,
    run_task_forecasting,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]  # .../src/bullfinch_forest_ml_demo/training/train_baselines.py
    data_path = project_root / "data" / "processed" / "trees_merged.parquet"

    print("Using dataset:", data_path)

    # ---- Task A: Classification ----
    cls_cfg = ClassificationConfig(data_path=data_path)
    cls_out = run_task_classification(cls_cfg)

    print("\n=== Task A: Health Classification (LogReg baseline) ===")
    print("Macro F1:", cls_out["macro_f1"])
    print("Train/Test:", cls_out["n_train"], "/", cls_out["n_test"])
    print("Features:", cls_out["features_n"])
    print("\nClassification report:\n", cls_out["classification_report"])

    # ---- Task B: Forecasting ----
    fc_cfg = ForecastingConfig(data_path=data_path)
    fc_out = run_task_forecasting(fc_cfg)

    print("\n=== Task B: Trunk Lean Forecasting (LinearRegression lag baseline) ===")
    print("MAE:", fc_out["mae"], " | RMSE:", fc_out["rmse"])
    print("Naive MAE:", fc_out["mae_naive"], " | Naive RMSE:", fc_out["rmse_naive"])
    print("Train/Test:", fc_out["n_train"], "/", fc_out["n_test"])
    print("Features:", fc_out["features_n"])
    print("Horizon days:", fc_out["horizon_days"])

    #TODO Next step later: save artifacts + MLflow logging.
    print("\nNext: add MLflow logging + artifact saving into /artifacts.")


if __name__ == "__main__":
    main()
