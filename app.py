import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="Apple Stock Forecast", page_icon="📈", layout="wide")
st.title("📈 Apple (AAPL) Stock Price Forecast")
st.warning("⚠️ **Disclaimer:** This is an educational machine learning project. It is NOT financial advice. Do not use this to make real trading decisions.")

# 2. Data Loading (Cached so it doesn't download every time)
@st.cache_data(ttl=3600) # Refreshes data every hour
def load_data():
    # Pull 5 years of live data from Yahoo Finance
    # multi_level_index=False prevents the MultiIndex column issue entirely
    df = yf.download("AAPL", period="5y", progress=False, multi_level_index=False)
    
    # Fallback flattening just in case
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    ts = df['Close'].dropna()
    
    # Force the index to be 'Business Days' and fill any gaps
    ts = ts.asfreq('B').ffill()
    
    return ts

# 3. Model Training (Cached so it only trains once)
@st.cache_resource
def train_model(ts, order=(2,1,2)):
    model = ARIMA(ts, order=order, trend='t')
    return model.fit()

# Sidebar Settings
st.sidebar.header("Settings")
horizon = st.sidebar.slider("Forecast Horizon (trading days)", min_value=5, max_value=60, value=30)

# Load Data and Train
with st.spinner("Fetching live data and training ARIMA model..."):
    ts = load_data()
    log_ts = np.log(ts)
    
    order = (2, 1, 2) 
    fit = train_model(log_ts, order)

# 4. Make Forecast
fc = fit.get_forecast(horizon)
future_idx = pd.bdate_range(ts.index[-1] + pd.Timedelta(days=1), periods=horizon)

forecast_mean = np.exp(fc.predicted_mean.values)

# CRITICAL FIX: Convert the confidence interval DataFrame to a numpy array
conf_int = np.exp(fc.conf_int(alpha=0.05).values)

# 5. Plotting with Plotly
fig = go.Figure()

# Plot last 250 days of actual data
fig.add_trace(go.Scatter(x=ts.index[-250:], y=ts.iloc[-250:], name="Actual Price", line=dict(color="black")))

# Plot Forecast
fig.add_trace(go.Scatter(x=future_idx, y=forecast_mean, name="Forecast", line=dict(color="crimson")))

# Plot Confidence Interval (using array slicing)
fig.add_trace(go.Scatter(x=future_idx, y=conf_int[:, 1], fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False))
fig.add_trace(go.Scatter(x=future_idx, y=conf_int[:, 0], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)', fillcolor='rgba(255,0,0,0.15)', name='95% Confidence Interval'))

fig.update_layout(
    title="Apple Stock Price: Last 250 Days & Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# 6. Show the Forecast Data Table
st.subheader("Forecast Data")
st.dataframe(pd.DataFrame({
    "Forecast": forecast_mean,
    "Lower 95%": conf_int[:, 0],
    "Upper 95%": conf_int[:, 1]
}, index=future_idx).round(2))
