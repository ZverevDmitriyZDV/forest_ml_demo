from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient


def project_root_from_file(file: str) -> Path:
    # .../src/bullfinch_forest_ml_demo/training/<file>.py
    return Path(file).resolve().parents[3]


def load_env(project_root: Path) -> None:

    env_file = os.getenv("ENV_FILE")
    if env_file:
        p = Path(env_file)
        if not p.is_absolute():
            p = project_root / p
        if not p.exists():
            raise RuntimeError(f"ENV_FILE is set but file not found: {p}")
        load_dotenv(p, override=True)
        return

    for name in (".env.local", ".env"):
        p = project_root / name
        if p.exists():
            load_dotenv(p, override=True)
            return


def setup_mlflow_or_die() -> MlflowClient:
    """
    MLflow server (http/https),we need Registry + aliases.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").strip()
    mlflow.set_tracking_uri(tracking_uri)

    uri = mlflow.get_tracking_uri()
    if not (uri.startswith("http://") or uri.startswith("https://")):
        raise RuntimeError(f"Expected MLflow server tracking URI (http/https), got: {uri}")

    client = MlflowClient()
    # быстрый пинг
    _ = client.search_experiments(max_results=1)
    return client


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)


def register_logged_model_and_set_alias(
    client: MlflowClient,
    logged_model_uri: str,
    registry_name: str,
    alias: str = "prod",
) -> int:
    """
    Register NOT via runs:/.../model, but via the logged model uri,
    which will return log_model(...).model_uri.
    """
    ensure_registered_model(client, registry_name)

    mv = mlflow.register_model(model_uri=logged_model_uri, name=registry_name)
    version = int(mv.version)

    client.set_registered_model_alias(registry_name, alias, version)
    return version


def print_links(run_id: str) -> None:
    uri = mlflow.get_tracking_uri().rstrip("/")
    print(f"View run at: {uri}/#/experiments/1/runs/{run_id}")
    print(f"View experiments at: {uri}/#/experiments")
