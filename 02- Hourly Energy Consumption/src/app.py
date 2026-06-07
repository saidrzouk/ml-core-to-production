from datetime import timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "AEP_hourly.csv"
MODEL_DIR = PROJECT_DIR / "models"
PROPHET_MODEL_PATH = MODEL_DIR / "prophet_model.pkl"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

WINDOW = 30
DEFAULT_HORIZON_OPTIONS = (30, 60, 90, 180)


st.set_page_config(page_title="Energy Demand Forecaster", page_icon="⚡", layout="wide")


@st.cache_resource
def load_models():
    prophet = joblib.load(PROPHET_MODEL_PATH)
    lstm = load_model(LSTM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return prophet, lstm, scaler


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"], index_col="Datetime")
    df.columns = ["y"]
    daily = df["y"].resample("D").mean().dropna().reset_index()
    daily.columns = ["ds", "y"]
    return daily


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


prophet_model, lstm_model, scaler = load_models()
daily = load_data()

with st.sidebar:
    st.title("Controls")
    st.subheader("Date range")

    min_date = daily["ds"].min().date()
    max_date = daily["ds"].max().date()
    default_start = max(min_date, max_date - timedelta(days=180))

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "From", value=default_start, min_value=min_date, max_value=max_date
        )
    with col2:
        end_date = st.date_input(
            "To", value=max_date, min_value=min_date, max_value=max_date
        )

    st.subheader("Forecast horizon")
    horizon = st.selectbox("Days ahead", DEFAULT_HORIZON_OPTIONS, index=2)

    st.subheader("Show models")
    show_prophet = st.toggle("Prophet", value=True)
    show_lstm = st.toggle("LSTM", value=True)
    show_naive = st.toggle("Naive baseline", value=False)
    show_ci = st.toggle("Confidence interval (Prophet)", value=True)

    run = st.button("Run forecast", type="primary", use_container_width=True)

st.title("Energy demand forecaster")
st.caption("Prophet vs LSTM — AEP hourly energy dataset")

if not run:
    st.info("Set your date range and options in the sidebar, then click Run forecast.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

mask = (daily["ds"].dt.date >= start_date) & (daily["ds"].dt.date <= end_date)
df_range = daily[mask].reset_index(drop=True)

test_size = max(30, int(len(df_range) * 0.2))
if len(df_range) <= WINDOW + test_size:
    st.error(f"Select more than {WINDOW + test_size} days of data.")
    st.stop()

last_date = df_range["ds"].max()

with st.spinner("Running Prophet..."):
    future = pd.DataFrame(
        {
            "ds": pd.date_range(
                start=last_date + timedelta(days=1), periods=horizon, freq="D"
            )
        }
    )
    prophet_fc = prophet_model.predict(future)

with st.spinner("Running LSTM..."):
    seed = scaler.transform(df_range[["y"]].tail(WINDOW))
    lstm_preds_scaled = []
    current_seq = seed.copy()

    for _ in range(horizon):
        x = current_seq.reshape(1, WINDOW, 1)
        pred = lstm_model.predict(x, verbose=0)[0][0]
        lstm_preds_scaled.append(pred)
        current_seq = np.vstack([current_seq[1:], [[pred]]])

    lstm_preds = scaler.inverse_transform(
        np.array(lstm_preds_scaled).reshape(-1, 1)
    ).flatten()
    lstm_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=horizon, freq="D"
    )

hist_test = df_range.tail(test_size).reset_index(drop=True)
hist_future = prophet_model.predict(hist_test[["ds"]])
prophet_hist_preds = hist_future["yhat"].values

full_scaled = scaler.transform(df_range[["y"]])
lstm_hist_preds = []
for i in range(len(df_range) - test_size - WINDOW, len(df_range) - test_size):
    x = full_scaled[i : i + WINDOW].reshape(1, WINDOW, 1)
    pred = lstm_model.predict(x, verbose=0)[0][0]
    lstm_hist_preds.append(pred)

lstm_hist_preds = scaler.inverse_transform(
    np.array(lstm_hist_preds).reshape(-1, 1)
).flatten()

y_true = hist_test["y"].values[: len(lstm_hist_preds)]
naive_p = df_range["y"].values[-(test_size + 1) : -1][: len(y_true)]

pm = metrics(y_true, prophet_hist_preds[: len(y_true)])
lm = metrics(y_true, lstm_hist_preds)
nm = metrics(y_true, naive_p)

st.subheader("Model performance on selected range")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Naive MAE", f"{nm[0]:,.0f} MW")
c2.metric(
    "Prophet MAE",
    f"{pm[0]:,.0f} MW",
    delta=f"{(nm[0] - pm[0]) / nm[0] * 100:.1f}% vs baseline",
)
c3.metric(
    "LSTM MAE",
    f"{lm[0]:,.0f} MW",
    delta=f"{(nm[0] - lm[0]) / nm[0] * 100:.1f}% vs baseline",
)
c4.metric("Naive MAPE", f"{nm[2]:.1f}%")
c5.metric("Prophet MAPE", f"{pm[2]:.1f}%")
c6.metric("LSTM MAPE", f"{lm[2]:.1f}%")

st.subheader(f"Forecast — next {horizon} days from {end_date}")

fig = go.Figure()
context = df_range.tail(90)
fig.add_trace(
    go.Scatter(
        x=context["ds"],
        y=context["y"],
        name="Actual (history)",
        line=dict(color="#888780", width=1.5),
        opacity=0.7,
    )
)

if show_prophet and show_ci:
    fig.add_trace(
        go.Scatter(
            x=pd.concat([prophet_fc["ds"], prophet_fc["ds"][::-1]]),
            y=pd.concat([prophet_fc["yhat_upper"], prophet_fc["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(55,138,221,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Prophet 95% CI",
            showlegend=True,
        )
    )

if show_prophet:
    fig.add_trace(
        go.Scatter(
            x=prophet_fc["ds"],
            y=prophet_fc["yhat"],
            name="Prophet",
            line=dict(color="#185FA5", width=2),
        )
    )

if show_lstm:
    fig.add_trace(
        go.Scatter(
            x=lstm_dates,
            y=lstm_preds,
            name="LSTM",
            line=dict(color="#A32D2D", width=2, dash="dot"),
        )
    )

if show_naive:
    naive_fc_val = df_range["y"].iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D"),
            y=[naive_fc_val] * horizon,
            name="Naive baseline",
            line=dict(color="#888780", width=1, dash="dash"),
        )
    )

fig.add_shape(
    type="line",
    x0=str(last_date),
    x1=str(last_date),
    y0=0,
    y1=1,
    xref="x",
    yref="paper",
    line=dict(color="#888780", width=1, dash="dash"),
    opacity=0.5,
)
fig.add_annotation(
    x=str(last_date),
    y=1,
    xref="x",
    yref="paper",
    text="Forecast starts",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    font=dict(size=11, color="#888780"),
)
fig.update_layout(
    height=420,
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis_title="Date",
    yaxis_title="Megawatts (MW)",
    hovermode="x unified",
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Residuals", "Prophet components", "Raw data"])

with tab1:
    st.caption("Error = Actual − Predicted over the historical test window")
    fig_r = go.Figure()
    fig_r.add_trace(
        go.Scatter(
            x=hist_test["ds"][: len(y_true)],
            y=y_true - prophet_hist_preds[: len(y_true)],
            name="Prophet residuals",
            line=dict(color="#185FA5"),
        )
    )
    fig_r.add_trace(
        go.Scatter(
            x=hist_test["ds"][: len(lstm_hist_preds)],
            y=y_true - lstm_hist_preds,
            name="LSTM residuals",
            line=dict(color="#A32D2D", dash="dot"),
        )
    )
    fig_r.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig_r.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig_r, use_container_width=True)
    st.caption("Residuals clustered around 0 = good. Systematic drift = model bias.")

with tab2:
    st.caption("What Prophet learned: trend, weekly pattern, yearly seasonality")
    prophet_fig = prophet_model.plot_components(prophet_model.predict(df_range[["ds"]]))
    st.pyplot(prophet_fig)

with tab3:
    st.caption("Raw daily values for the selected date range")
    st.dataframe(
        df_range.rename(columns={"ds": "Date", "y": "Energy (MW)"})
        .set_index("Date")
        .style.format({"Energy (MW)": "{:,.0f}"}),
        use_container_width=True,
        height=300,
    )
    csv = df_range.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "aep_selected_range.csv", "text/csv")
