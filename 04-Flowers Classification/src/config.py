import os
from pathlib import Path


def _path_env(name, default):
    value = os.getenv(name)
    return Path(value) if value else default

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = _path_env("FLOWERS_DATA_DIR", PROJECT_DIR / "data" / "raw" / "flowers")
MODEL_PATH = _path_env("FLOWERS_MODEL_PATH", PROJECT_DIR / "models" / "flower_classifier_model.keras")
CLASS_NAMES_PATH = _path_env("FLOWERS_CLASS_NAMES_PATH", PROJECT_DIR / "models" / "class_names.json")
IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv("FLOWERS_BATCH_SIZE", "32"))
SEED = int(os.getenv("FLOWERS_SEED", "42"))
EPOCHS = int(os.getenv("FLOWERS_EPOCHS", "10"))
FINE_TUNE_EPOCHS = int(os.getenv("FLOWERS_FINE_TUNE_EPOCHS", "5"))
