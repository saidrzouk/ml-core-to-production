# Hourly Energy Consumption Forecasting

Forecasts AEP energy demand using Prophet, an LSTM model, and a naive baseline in a Streamlit dashboard.

## Structure

- `data/raw/AEP_hourly.csv` — local raw hourly energy dataset.
- `src/config.py` — shared paths and constants; environment variables can override default locations.
- `src/app.py` — Streamlit forecasting dashboard.
- `scripts/train_models.py` — Prophet and LSTM training script.
- `models/` — saved Prophet, LSTM, and scaler artifacts.
- `notebooks/` — exploratory notebook.
- `reports/` — generated figures and HTML outputs.

## Configuration

Use these environment variables to override default data and model paths:

- `ENERGY_DATA_PATH` — CSV input data path.
- `ENERGY_MODEL_DIR` — output directory for saved models and scaler.
- `ENERGY_PROPHET_MODEL_PATH` — saved Prophet model file path.
- `ENERGY_LSTM_MODEL_PATH` — saved LSTM model file path.
- `ENERGY_SCALER_PATH` — saved scaler file path.

Defaults are based on the local project layout under `data/raw` and `models/`.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train_models.py
```

## Run

```bash
streamlit run src/app.py
```


