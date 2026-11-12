
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Apple Stock Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Apple Stock Price Forecast")
st.write("Forecast Apple stock prices using the ARIMA time-series model.")

st.warning(
    "Educational project only. This is not financial advice."
)


@st.cache_data(ttl=3600)
def load_data():
    df = yf.download(
        "AAPL",
        period="5y",
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df["Close"].dropna()


@st.cache_resource
def train_model(ts):
    model = ARIMA(
        ts,
        order=(2, 1, 2),
        trend="t"
    )
    return model.fit()


st.sidebar.header("Settings")

horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=5,
    max_value=60,
    value=30
)

with st.spinner("Fetching stock data and training ARIMA model..."):

    try:
        ts = load_data()

        if ts.empty:
            st.error("No stock data received. Please try again.")
            st.stop()

        log_ts = np.log(ts)

        fit = train_model(log_ts)

        fc = fit.get_forecast(steps=horizon)

        future_idx = pd.bdate_range(
            ts.index[-1] + pd.Timedelta(days=1),
            periods=horizon
        )

        forecast_mean = np.exp(
            np.asarray(fc.predicted_mean)
        )

        conf_int = np.exp(
            fc.conf_int(alpha=0.05).to_numpy()
        )

    except Exception as e:
        st.error(f"Application error: {e}")
        st.stop()


# Forecast chart

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ts.index[-250:],
    y=ts.iloc[-250:],
    name="Actual Price",
    line=dict(color="black")
))

fig.add_trace(go.Scatter(
    x=future_idx,
    y=forecast_mean,
    name="Forecast",
    line=dict(color="crimson")
))

fig.add_trace(go.Scatter(
    x=future_idx,
    y=conf_int[:, 1],
    mode="lines",
    line=dict(color="rgba(255,0,0,0)"),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=future_idx,
    y=conf_int[:, 0],
    mode="lines",
    fill="tonexty",
    fillcolor="rgba(255,0,0,0.15)",
    line=dict(color="rgba(255,0,0,0)"),
    name="95% Confidence Interval"
))

fig.update_layout(
    title="Apple Stock Price Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast Data")

forecast_df = pd.DataFrame(
    {
        "Forecast": forecast_mean,
        "Lower 95%": conf_int[:, 0],
        "Upper 95%": conf_int[:, 1]
    },
    index=future_idx
).round(2)

st.dataframe(forecast_df, use_container_width=True)

st.caption(
    "Source: Yahoo Finance | Model: ARIMA(2,1,2) | "
    "Forecast intervals: 95%"
)
