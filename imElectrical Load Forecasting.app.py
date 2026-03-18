# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda. All rights reserved.
# Proprietary and Confidential.
# ==============================================================================

import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import datetime
import numpy as np

st.set_page_config(page_title="PowerGuard AI - Global Load Forecasting", page_icon="🌍", layout="wide")

# ==========================================
# --- 1. Login System ---
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #007acc;'>🌍 PowerGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Global Capacity Planning & Load Forecasting</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Secure Login", use_container_width=True):
            if username == "admin" and password == "DBA2026": 
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("❌ Invalid Username or Password")
    st.stop()

# ==========================================
# --- 2. Global Engineering Database ---
# ==========================================
# Base Load (MW), Cooling Factor (alpha), Heating Factor (beta) based on country profile
GLOBAL_GRID_CONFIG = {
    "Africa": {
        "Egypt": {"lat": 26.82, "lon": 30.80, "base": 25000, "cool_k": 850, "heat_k": 80},
        "South Africa": {"lat": -30.55, "lon": 22.93, "base": 30000, "cool_k": 200, "heat_k": 600},
        "Morocco": {"lat": 31.79, "lon": -7.09, "base": 15000, "cool_k": 300, "heat_k": 150},
        "Nigeria": {"lat": 9.08, "lon": 8.67, "base": 12000, "cool_k": 400, "heat_k": 10},
        "Kenya": {"lat": -1.29, "lon": 36.82, "base": 8000, "cool_k": 150, "heat_k": 50}
    },
    "Asia": {
        "Saudi Arabia": {"lat": 23.88, "lon": 45.07, "base": 45000, "cool_k": 1500, "heat_k": 50},
        "UAE": {"lat": 23.42, "lon": 53.84, "base": 20000, "cool_k": 1200, "heat_k": 20},
        "India": {"lat": 20.59, "lon": 78.96, "base": 160000, "cool_k": 2500, "heat_k": 100},
        "Japan": {"lat": 36.20, "lon": 138.25, "base": 90000, "cool_k": 800, "heat_k": 1100},
        "China": {"lat": 35.86, "lon": 104.19, "base": 500000, "cool_k": 4000, "heat_k": 5000}
    },
    "Europe": {
        "Germany": {"lat": 51.16, "lon": 10.45, "base": 55000, "cool_k": 150, "heat_k": 1200},
        "France": {"lat": 46.22, "lon": 2.21, "base": 50000, "cool_k": 200, "heat_k": 1400}, # Heavy electric heating
        "UK": {"lat": 55.37, "lon": -3.43, "base": 35000, "cool_k": 100, "heat_k": 900},
        "Italy": {"lat": 41.87, "lon": 12.56, "base": 40000, "cool_k": 500, "heat_k": 600},
        "Spain": {"lat": 40.46, "lon": -3.74, "base": 30000, "cool_k": 700, "heat_k": 400}
    },
    "North America": {
        "USA": {"lat": 37.09, "lon": -95.71, "base": 450000, "cool_k": 4500, "heat_k": 4000},
        "Canada": {"lat": 56.13, "lon": -106.34, "base": 70000, "cool_k": 100, "heat_k": 2500},
        "Mexico": {"lat": 23.63, "lon": -102.55, "base": 40000, "cool_k": 900, "heat_k": 100}
    }
}

# Scientific Equation: Temperature to Load Conversion (CDD & HDD)
def calculate_thermodynamic_load(temp, base, cool_k, heat_k):
    if temp > 22: # Cooling Degree Days (CDD) impact
        return base + (cool_k * ((temp - 22) ** 1.3))
    elif temp < 15: # Heating Degree Days (HDD) impact
        return base + (heat_k * ((15 - temp) ** 1.2))
    else: # Thermal comfort zone (Base Load only)
        return base

# Fetch historical weather & generate load
@st.cache_data
def get_country_data(country_name, config):
    lat = config['lat']
    lon = config['lon']
    base = config['base']
    cool_k = config['cool_k']
    heat_k = config['heat_k']
    
    # Fetch 5 years of historical temperature from Open-Meteo
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2018-01-01&end_date=2023-12-31&daily=temperature_2m_mean&timezone=auto"
    response = requests.get(url).json()
    
    dates = response['daily']['time']
    temps = response['daily']['temperature_2m_mean']
    
    df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Temp_C': temps})
    df.dropna(inplace=True)
    
    # Apply scientific equation to generate load + add 3% random noise for realism
    df['Load_MW'] = df['Temp_C'].apply(lambda t: calculate_thermodynamic_load(t, base, cool_k, heat_k))
    df['Load_MW'] = df['Load_MW'] * np.random.uniform(0.97, 1.03, len(df))
    
    return df[['Date', 'Load_MW']].rename(columns={'Date': 'ds', 'Load_MW': 'y'})

@st.cache_resource
def train_and_forecast(_df):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(_df)
    future = model.make_future_dataframe(periods=6000) # Forecast to ~2040
    forecast = model.predict(future)
    return forecast

# ==========================================
# --- 3. Main Dashboard UI ---
# ==========================================
st.sidebar.markdown("### 👤 User Profile")
st.sidebar.markdown("**Eng. Mahmoud Reda**\n\n*DBA Candidate*")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state['logged_in'] = False
    st.rerun()
st.sidebar.markdown("---")

st.title("🌍 Global Electrical Load Forecasting")
st.markdown("AI-Driven Capacity Planning utilizing thermodynamic principles (Heating/Cooling Degree Days).")

# 1. Geographic Selection
st.sidebar.header("📍 Select Region")
selected_continent = st.sidebar.selectbox("Continent:", list(GLOBAL_GRID_CONFIG.keys()))
selected_country = st.sidebar.selectbox("Country:", list(GLOBAL_GRID_CONFIG[selected_continent].keys()))

country_config = GLOBAL_GRID_CONFIG[selected_continent][selected_country]

# Execute Data Fetch & Model Training
with st.spinner(f"🛰️ Fetching Satellite Weather Data & Training AI for {selected_country}..."):
    df_historical = get_country_data(selected_country, country_config)
    forecast = train_and_forecast(df_historical)

# 2. Date Selection
st.sidebar.header("📅 Select Target Date")
min_date = forecast['ds'].dt.date.values[-6000] # Start of forecast
max_date = forecast['ds'].dt.date.values[-1]
selected_date = st.sidebar.date_input("Target Date (Up to 2040):", min_value=min_date, max_value=max_date, value=min_date + datetime.timedelta(days=365))

# Filter Results
prediction_row = forecast[forecast['ds'].dt.date == selected_date]

if not prediction_row.empty:
    pred_load = prediction_row['yhat'].values[0]
    trend_val = prediction_row['trend'].values[0]

    # KPIs
    st.markdown(f"### 🎯 Grid Load Projections for **{selected_country}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ Predicted Load", f"{pred_load:,.0f} MW", f"Based on {selected_country} climate profile")
    col2.metric("📊 Underlying Trend (Base Load)", f"{trend_val:,.0f} MW", "Long-term capacity need")
    col3.metric("🌡️ Grid Nature", "Summer-Peaking" if country_config['cool_k'] > country_config['heat_k'] else "Winter-Peaking")
    st.markdown("---")

    # Charts
    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        st.markdown(f"#### 📈 Long-Term Forecast (To 2040) - {selected_country}")
        forecast['Year'] = forecast['ds'].dt.year
        yearly_data = forecast.groupby('Year')['yhat'].mean().reset_index()
        fig_trend = px.line(yearly_data, x='Year', y='yhat', markers=True, line_shape='spline')
        fig_trend.update_traces(line_color='#007acc', line_width=3)
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### 🌦️ Seasonality (Summer vs Winter Load)")
        forecast['Month'] = forecast['ds'].dt.month_name()
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_data = forecast.groupby('Month')['yearly'].mean().reset_index()
        monthly_data['Month'] = pd.Categorical(monthly_data['Month'], categories=months_order, ordered=True)
        monthly_data = monthly_data.sort_values('Month')
        fig_season = px.bar(monthly_data, x='Month', y='yearly', color='yearly', color_continuous_scale='RdBu_r')
        fig_season.update_layout(showlegend=False)
        st.plotly_chart(fig_season, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>© 2026 PowerGuard AI by Eng. Mahmoud Reda. All rights reserved.</div>", unsafe_allow_html=True)