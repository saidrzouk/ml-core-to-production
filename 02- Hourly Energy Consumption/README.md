# Hourly Energy Consumption Forecasting

Forecasts AEP energy demand using Prophet, an LSTM model, and a naive baseline in a Streamlit dashboard.

## Structure

- `data/raw/AEP_hourly.csv` — local raw hourly energy dataset.
- `src/app.py` — Streamlit forecasting dashboard.
- `src/config.py` — shared paths and constants.
- `scripts/train_models.py` — Prophet and LSTM training script.
- `models/` — saved Prophet, LSTM, and scaler artifacts.
- `notebooks/` — exploratory notebook.
- `reports/` — generated figures and HTML outputs.

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


