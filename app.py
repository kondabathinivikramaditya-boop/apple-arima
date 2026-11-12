import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Apple Stock Forecast", page_icon="📈", layout="wide")
st.title("📈 Apple (AAPL) Stock Price Forecast")
st.warning("⚠️ **Disclaimer:** This is an educational machine learning project. It is NOT financial advice.")

@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("AAPL", period="5y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df['Close'].dropna()

@st.cache_resource
def train_model(ts, order=(2,1,2)):
    model = ARIMA(ts, order=order, trend='t')
    return model.fit()

st.sidebar.header("Settings")
horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=5, max_value=60, value=30)

with st.spinner("Fetching data and training model..."):
    ts = load_data()
    log_ts = np.log(ts)
    order = (2, 1, 2) 
    fit = train_model(log_ts, order)

fc = fit.get_forecast(horizon)
future_idx = pd.bdate_range(ts.index[-1] + pd.Timedelta(days=1), periods=horizon)
forecast_mean = np.exp(fc.predicted_mean.values)
conf_int = np.exp(fc.conf_int(alpha=0.05))

fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index[-250:], y=ts.iloc[-250:], name="Actual Price", line=dict(color="black")))
fig.add_trace(go.Scatter(x=future_idx, y=forecast_mean, name="Forecast", line=dict(color="crimson")))
fig.add_trace(go.Scatter(x=future_idx, y=conf_int[:, 1], fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False))
fig.add_trace(go.Scatter(x=future_idx, y=conf_int[:, 0], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)', fillcolor='rgba(255,0,0,0.15)', name='95% Confidence Interval'))

fig.update_layout(title="Apple Stock Price: Forecast", xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified", height=600)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast Data")
st.dataframe(pd.DataFrame({"Forecast": forecast_mean, "Lower 95%": conf_int[:, 0], "Upper 95%": conf_int[:, 1]}, index=future_idx).round(2))