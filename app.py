
    st.stop()


# ---------------------------------------------------------
# Data status
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Latest available price (USD)", f"${ts.iloc[-1]:,.2f}")
col2.metric("Historical observations", f"{len(ts):,}")
col3.metric("Forecast horizon", f"{horizon} trading days")

st.caption(
    f"Data source: {data_source} | "
    f"Historical period: {ts.index.min().date()} to {ts.index.max().date()}"
)

if fallback_note:
    st.info(fallback_note)


# ---------------------------------------------------------
# Forecast chart
# ---------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ts.index[-250:],
    y=ts.iloc[-250:],
    name="Actual Price",
    line=dict(color="black")
))

fig.add_trace(go.Scatter(
    x=future_idx,
    y=conf_int[:, 1],
    mode="lines",
    line=dict(color="rgba(255,0,0,0)"),
    showlegend=False,
    hoverinfo="skip"
))

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

fig.add_trace(go.Scatter(
    x=future_idx,
    y=forecast_mean,
    name="Forecast",
    line=dict(color="crimson", width=2)
))

fig.update_layout(
    title="Apple Stock Price Forecast",
    xaxis_title="Date",
    yaxis_title="Adjusted Price (USD)",
    hovermode="x unified",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Forecast table
# ---------------------------------------------------------
st.subheader("Forecast Data")

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

st.dataframe(forecast_df, use_container_width=True)

csv_bytes = forecast_df.to_csv().encode("utf-8")
st.download_button(
    "Download forecast CSV",
    data=csv_bytes,
    file_name="apple_stock_forecast.csv",
    mime="text/csv"
)

st.caption(
    "Model: ARIMA(2, 1, 2) with trend='t' on log prices. "
    "Forecast intervals: 95%. Future dates are business weekdays; "
    "they do not account for exchange holidays."
)
