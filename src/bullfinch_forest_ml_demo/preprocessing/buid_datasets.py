"""
Build processed datasets from raw CSV files.

Expected inputs (default):
- data/raw/trees_level1.csv
- data/raw/trees_daily_dataset.csv

Outputs (default):
- data/processed/trees_level1.parquet
- data/processed/trees_daily.parquet
- data/processed/trees_merged.parquet

Run:
  python -m bullfinch_forest_ml_demo.preprocessing.buid_datasets

Optional:
  set environment variables (or adjust dataclass defaults):
    BULLFINCH_RAW_DIR=data/raw
    BULLFINCH_PROCESSED_DIR=data/processed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import pandas as pd

# ---- Project root discovery (safe for src-layout) ----
# this file: .../src/bullfinch_forest_ml_demo/preprocessing/buid_datasets.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DataPaths:
    raw_dir: str = os.getenv("BULLFINCH_RAW_DIR", "data/raw")
    processed_dir: str = os.getenv("BULLFINCH_PROCESSED_DIR", "data/processed")


@dataclass(frozen=True)
class PreprocessConfig:
    data: DataPaths = DataPaths()
    level1_filename: str = "trees_level1.csv"
    daily_filename: str = "trees_daily_dataset.csv"

    # output filenames
    out_level1: str = "trees_level1.parquet"
    out_daily: str = "trees_daily.parquet"
    out_merged: str = "trees_merged.parquet"


def _abs_dir(rel_or_abs: str) -> Path:
    """Resolve config path relative to PROJECT_ROOT unless absolute."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n\n"
            f"Expected raw datasets in: {path.parent}\n"
            f"Fix:\n"
            f"  1) Run dataset generator to create raw CSVs\n"
            f"  2) Ensure they are copied into data/raw/\n"
        )


def _read_inputs(cfg: PreprocessConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = _abs_dir(cfg.data.raw_dir)
    level1_path = raw_dir / cfg.level1_filename
    daily_path = raw_dir / cfg.daily_filename

    _require_file(level1_path)
    _require_file(daily_path)

    l1 = pd.read_csv(level1_path)
    d = pd.read_csv(daily_path, parse_dates=["timestamp"])

    return l1, d


def _clean_level1(l1: pd.DataFrame) -> pd.DataFrame:
    # basic sanity
    if "tree_id" not in l1.columns:
        raise ValueError("Level1 dataset must contain 'tree_id' column")

    # strip strings
    for col in ["tree_id", "species", "location_zone", "forest_type", "soil_type"]:
        if col in l1.columns:
            l1[col] = l1[col].astype(str).str.strip()

    # types
    if "planting_year" in l1.columns:
        l1["planting_year"] = pd.to_numeric(l1["planting_year"], errors="coerce").astype("Int64")

    if "wind_exposure" in l1.columns:
        l1["wind_exposure"] = pd.to_numeric(l1["wind_exposure"], errors="coerce")

    # dedupe: keep first
    l1 = l1.drop_duplicates(subset=["tree_id"], keep="first").reset_index(drop=True)

    return l1


def _clean_daily(d: pd.DataFrame) -> pd.DataFrame:
    required = {"tree_id", "timestamp"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Daily dataset missing required columns: {sorted(missing)}")

    d["tree_id"] = d["tree_id"].astype(str).str.strip()

    # normalize sensor_status
    if "sensor_status" in d.columns:
        d["sensor_status"] = d["sensor_status"].astype(str).str.strip().str.lower()
        d.loc[~d["sensor_status"].isin(["ok", "degraded", "offline"]), "sensor_status"] = "ok"

    # numeric cols (safe coercion)
    num_cols = [
        "temperature",
        "humidity",
        "moisture_level",
        "sap_flow_rate",
        "leaf_color_index",
        "trunk_deg",
        "estimated_age",
        "biomass",
        "wind_exposure",
    ]
    for c in num_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # boolean
    if "risk_flag" in d.columns:
        # allow True/False, 0/1, strings
        d["risk_flag"] = d["risk_flag"].astype(str).str.lower().isin(["true", "1", "yes", "y"])

    # drop duplicates per (tree_id, timestamp)
    d = d.sort_values(["tree_id", "timestamp"]).drop_duplicates(["tree_id", "timestamp"], keep="last")

    return d.reset_index(drop=True)


def _merge(l1: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    # left join daily to level1 so we keep all daily rows
    merged = d.merge(l1, on="tree_id", how="left", suffixes=("", "_l1"))

    # quick check: how many daily rows lost static features
    if merged["species"].isna().any():
        # not fatal, but useful warning
        missing_n = int(merged["species"].isna().sum())
        print(f"[WARN] {missing_n} daily rows have no matching tree_id in level1.")

    return merged


def _write_outputs(cfg: PreprocessConfig, l1: pd.DataFrame, d: pd.DataFrame, merged: pd.DataFrame) -> None:
    out_dir = _abs_dir(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / cfg.out_level1).write_bytes(b"")  # touch early to reveal permission issues fast
    (out_dir / cfg.out_daily).write_bytes(b"")
    (out_dir / cfg.out_merged).write_bytes(b"")

    # write parquet
    l1.to_parquet(out_dir / cfg.out_level1, index=False)
    d.to_parquet(out_dir / cfg.out_daily, index=False)
    merged.to_parquet(out_dir / cfg.out_merged, index=False)

    # TODO: artifacts/preprocessing_report.json (краткий отчёт: пропуски, дубли, строки до/после)

    print("[OK] Wrote:")
    print(" -", out_dir / cfg.out_level1)
    print(" -", out_dir / cfg.out_daily)
    print(" -", out_dir / cfg.out_merged)


def run(cfg: PreprocessConfig | None = None) -> None:
    cfg = cfg or PreprocessConfig()

    l1, d = _read_inputs(cfg)

    l1 = _clean_level1(l1)
    d = _clean_daily(d)
    merged = _merge(l1, d)

    _write_outputs(cfg, l1, d, merged)


if __name__ == "__main__":
    run()


