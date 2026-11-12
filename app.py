
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from pandas.tseries.offsets import BDay
from statsmodels.tsa.arima.model import ARIMA


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Apple Stock Price Forecast",
    page_icon="📈",
    layout="wide"
)


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------
st.title("📈 Apple Stock Price Forecast")

st.write(
    "Forecast Apple stock prices using the ARIMA time-series model."
)

st.warning(
    "Educational project only. This is not financial advice. "
    "Historical dataset ends in December 2019."
)


# ---------------------------------------------------------
# Forecast settings
# ---------------------------------------------------------
st.sidebar.header("Settings")

horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=5,
    max_value=60,
    value=30
)


# ---------------------------------------------------------
# Load historical dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():

    file_path = Path(__file__).parent / "Dataset.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            "Dataset.csv was not found. "
            "Upload Dataset.csv beside app.py in GitHub."
        )

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    required_columns = ["Date", "Adj Close"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Adj Close"] = pd.to_numeric(
        df["Adj Close"],
        errors="coerce"
    )

    df = df.dropna(subset=["Date", "Adj Close"])

    df = df[df["Adj Close"] > 0]

    df = df.sort_values("Date")

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    df = df.set_index("Date")

    return df["Adj Close"].astype(float)


# ---------------------------------------------------------
# Load data and train ARIMA model
# ---------------------------------------------------------
try:

    with st.spinner(
        "Loading historical data and training ARIMA model..."
    ):

        ts = load_data()

        if len(ts) < 100:
            st.error(
                "Not enough historical observations "
                "to train the ARIMA model."
            )
            st.stop()

        # Log transformation
        log_ts = np.log(ts)

        # ARIMA model
        model = ARIMA(
            log_ts,
            order=(2, 1, 2),
            trend="t",
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit()

        # Forecast
        forecast_result = fitted_model.get_forecast(
            steps=horizon
        )

        forecast_log = np.asarray(
            forecast_result.predicted_mean
        )

        conf_log = np.asarray(
            forecast_result.conf_int(alpha=0.05)
        )

        # Convert log prices back to original prices
        forecast_mean = np.exp(forecast_log)

        conf_int = np.exp(conf_log)

        # Future business dates
        future_idx = pd.bdate_range(
            start=ts.index[-1] + BDay(1),
            periods=horizon
        )

except Exception as e:

    st.error(f"Application error: {e}")
    st.stop()


# ---------------------------------------------------------
# Data status
# ---------------------------------------------------------
st.success("ARIMA model trained successfully!")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest available price (USD)",
    f"${ts.iloc[-1]:,.2f}"
)

col2.metric(
    "Historical observations",
    f"{len(ts):,}"
)

col3.metric(
    "Forecast horizon",
    f"{horizon} trading days"
)

st.caption(
    f"Data source: Local historical CSV | "
    f"Historical period: {ts.index.min().date()} "
    f"to {ts.index.max().date()}"
)


# ---------------------------------------------------------
# Forecast chart
# ---------------------------------------------------------
fig = go.Figure()

# Actual historical prices
fig.add_trace(go.Scatter(
    x=ts.index[-250:],
    y=ts.iloc[-250:],
    name="Actual Price",
    line=dict(color="black", width=1.5)
))

# Upper confidence interval
fig.add_trace(go.Scatter(
    x=future_idx,
    y=conf_int[:, 1],
    mode="lines",
    line=dict(color="rgba(255,0,0,0)"),
    showlegend=False,
    hoverinfo="skip"
))

# Lower confidence interval
fig.add_trace(go.Scatter(
    x=future_idx,
    y=conf_int[:, 0],
    mode="lines",
    fill="tonexty",
    fillcolor="rgba(255,0,0,0.15)",
    line=dict(color="rgba(255,0,0,0)"),
    name="95% Confidence Interval",
    hoverinfo="skip"
))

# Forecast line
fig.add_trace(go.Scatter(
    x=future_idx,
    y=forecast_mean,
    name="Forecast",
    line=dict(color="crimson", width=2.5)
))

fig.update_layout(
    title="Apple Stock Price Forecast",
    xaxis_title="Date",
    yaxis_title="Adjusted Price (USD)",
    hovermode="x unified",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# Forecast table
# ---------------------------------------------------------
st.subheader("📊 Forecast Data")

forecast_df = pd.DataFrame(
    {
        "Forecast (USD)": forecast_mean,
        "Lower 95% (USD)": conf_int[:, 0],
        "Upper 95% (USD)": conf_int[:, 1]
    },
    index=future_idx
)

forecast_df.index.name = "Date"

forecast_df = forecast_df.round(2)

st.dataframe(
    forecast_df,
    use_container_width=True
)


# ---------------------------------------------------------
# Download forecast
# ---------------------------------------------------------
csv_bytes = forecast_df.to_csv().encode("utf-8")

st.download_button(
    label="📥 Download Forecast CSV",
    data=csv_bytes,
    file_name="apple_stock_forecast.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------
st.divider()

st.subheader("Model Information")

st.write("""
- **Algorithm:** ARIMA
- **Model Order:** (2, 1, 2)
- **Transformation:** Log transformation
- **Forecast Interval:** 95%
- **Historical Data:** Apple stock dataset (2012–2019)
- **Forecast Horizon:** User selectable (5–60 business days)
""")

st.caption(
    "Future dates represent business weekdays and do not "
    "account for exchange holidays. Forecasts are based "
    "on historical data and are not predictions of "
    "guaranteed future market prices."
)
