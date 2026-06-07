import os
from pathlib import Path


def _path_env(name, default):
    value = os.getenv(name)
    return Path(value) if value else default

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = _path_env("BOOKS_DATA_DIR", PROJECT_DIR / "data")
ARTIFACT_DIR = _path_env("BOOKS_ARTIFACT_DIR", PROJECT_DIR / "artifacts")
ARTIFACT_PATH = _path_env("BOOKS_ARTIFACT_PATH", ARTIFACT_DIR / "book_recommendation_artifact.joblib")
