import os
from pathlib import Path


def _path_env(name, default):
    value = os.getenv(name)
    return Path(value) if value else default

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = _path_env("ENERGY_DATA_PATH", PROJECT_DIR / "data" / "raw" / "AEP_hourly.csv")
MODEL_DIR = _path_env("ENERGY_MODEL_DIR", PROJECT_DIR / "models")
PROPHET_MODEL_PATH = _path_env("ENERGY_PROPHET_MODEL_PATH", MODEL_DIR / "prophet_model.pkl")
LSTM_MODEL_PATH = _path_env("ENERGY_LSTM_MODEL_PATH", MODEL_DIR / "lstm_model.keras")
SCALER_PATH = _path_env("ENERGY_SCALER_PATH", MODEL_DIR / "scaler.pkl")
WINDOW = int(os.getenv("ENERGY_WINDOW", "30"))
DEFAULT_HORIZON_OPTIONS = tuple(int(x) for x in os.getenv("ENERGY_DEFAULT_HORIZON_OPTIONS", "30,60,90,180").split(","))
