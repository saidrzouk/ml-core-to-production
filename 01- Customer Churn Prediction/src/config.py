import os
from pathlib import Path


def _path_env(name, default):
    value = os.getenv(name)
    return Path(value) if value else default

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = _path_env("CHURN_ARTIFACT_DIR", PROJECT_DIR / "artifacts" / "models")
MODEL_PATH = _path_env("CHURN_MODEL_PATH", ARTIFACT_DIR / "model.pkl")
PREPROCESSOR_PATH = _path_env("CHURN_PREPROCESSOR_PATH", ARTIFACT_DIR / "preprocessor.pkl")
