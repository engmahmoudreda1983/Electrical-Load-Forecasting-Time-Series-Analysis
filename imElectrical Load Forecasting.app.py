import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
import plotly.graph_objs as go
import datetime

# 1. Page Configuration
st.set_page_config(page_title="AI Load Forecasting", page_icon="⚡", layout="wide")

st.title("⚡ AI-Driven Electrical Load Forecasting")
st.markdown("Strategic Predictive Analytics for Power Grid Operations (1-Year Horizon).")

# 2. Load the Pre-trained Model (Fast Loading)
@st.cache_resource
def load_model():
    with open('prophet_model.json', 'r') as fin:
        model = model_from_json(fin.read())
    return model

model = load_model()

# 3. Generate Forecast
@st.cache_data
def get_forecast():
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast

forecast = get_forecast()

# 4. Sidebar for Interactivity
st.sidebar.header("📅 Select Forecast Date")
future_dates = forecast['ds'].dt.date.tail(365).values

selected_date = st.sidebar.date_input(
    "Choose a future date to predict load:", 
    min_value=future_dates[0], 
    max_value=future_dates[-1], 
    value=future_dates[0]
)

# 5. Filter Results
prediction_row = forecast[forecast['ds'].dt.date == selected_date]

if not prediction_row.empty:
    pred_load = prediction_row['yhat'].values[0]
    pred_min = prediction_row['yhat_lower'].values[0]
    pred_max = prediction_row['yhat_upper'].values[0]

    # Display KPI Metrics
    st.markdown("### 🎯 Expected Grid Load for Selected Date")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Load (MW)", f"{pred_load:,.2f} MW")
    col2.metric("Minimum Expected", f"{pred_min:,.2f} MW")
    col3.metric("Maximum Expected", f"{pred_max:,.2f} MW")

    # 6. Interactive Plotly Chart
    st.markdown("### 📈 Load Trend Analysis (Forecast)")
    fig = go.Figure()
    
    # Plot 365 days of predicted data
    future_forecast = forecast.tail(365)
    fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Predicted Load', line=dict(color='blue', width=2)))
    
    # Highlight selected date
    fig.add_trace(go.Scatter(x=[selected_date], y=[pred_load], mode='markers', name='Selected Target', marker=dict(color='red', size=14, symbol='star')))

    fig.update_layout(xaxis_title="Date", yaxis_title="Electrical Load (MW)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)