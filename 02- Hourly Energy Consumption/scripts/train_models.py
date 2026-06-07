import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.config import (  # noqa: E402
    DATA_PATH,
    LSTM_MODEL_PATH,
    MODEL_DIR,
    PROPHET_MODEL_PATH,
    SCALER_PATH,
    WINDOW,
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'], index_col='Datetime')
df.columns = ['y']
daily = df['y'].resample('D').mean().dropna().reset_index()
daily.columns = ['ds', 'y']

split = int(len(daily) * 0.8)
train = daily.iloc[:split]

print("Training Prophet...")
prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                  daily_seasonality=False, changepoint_prior_scale=0.05)
prophet.fit(train)
joblib.dump(prophet, PROPHET_MODEL_PATH)
print("Prophet saved.")

print("Training LSTM...")
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[['y']])
joblib.dump(scaler, SCALER_PATH)

def make_sequences(data, window=WINDOW):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled, WINDOW)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_split=0.1, callbacks=[es], verbose=1)
model.save(LSTM_MODEL_PATH)
print("LSTM saved.")
print("Training complete. Run: streamlit run src/app.py")
