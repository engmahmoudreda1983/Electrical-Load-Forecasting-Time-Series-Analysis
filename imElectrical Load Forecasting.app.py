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

st.set_page_config(page_title="PowerGuard AI - Global Load", page_icon="🌍", layout="wide")

# ==========================================
# --- 1. Login System ---
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #007acc;'>🌍 PowerGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Global Capacity Planning & Load Forecasting</h3>", unsafe_allow_html=True)
    
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
GLOBAL_GRID_CONFIG = {
    "Africa": {
        "Egypt": {"lat": 30.04, "lon": 31.23, "base": 25000, "cool_k": 1200, "heat_k": 50, "growth": 0.035}, 
        "South Africa": {"lat": -26.20, "lon": 28.04, "base": 30000, "cool_k": 200, "heat_k": 800, "growth": 0.015}, 
        "Morocco": {"lat": 33.57, "lon": -7.58, "base": 15000, "cool_k": 350, "heat_k": 150, "growth": 0.04},
        "Nigeria": {"lat": 9.08, "lon": 8.67, "base": 12000, "cool_k": 400, "heat_k": 10, "growth": 0.05},
        "Kenya": {"lat": -1.29, "lon": 36.82, "base": 8000, "cool_k": 150, "heat_k": 50, "growth": 0.045}
    },
    "Asia": {
        "Saudi Arabia": {"lat": 24.71, "lon": 46.67, "base": 45000, "cool_k": 1800, "heat_k": 20, "growth": 0.04},
        "UAE": {"lat": 25.20, "lon": 55.27, "base": 20000, "cool_k": 1500, "heat_k": 10, "growth": 0.03},
        "India": {"lat": 28.61, "lon": 77.20, "base": 160000, "cool_k": 2500, "heat_k": 100, "growth": 0.06},
        "Japan": {"lat": 35.67, "lon": 139.65, "base": 90000, "cool_k": 800, "heat_k": 1500, "growth": 0.002},
        "China": {"lat": 39.90, "lon": 116.40, "base": 500000, "cool_k": 4000, "heat_k": 5000, "growth": 0.05}
    },
    "Europe": {
        "Germany": {"lat": 52.52, "lon": 13.40, "base": 55000, "cool_k": 50, "heat_k": 3500, "growth": 0.005}, 
        "France": {"lat": 48.85, "lon": 2.35, "base": 50000, "cool_k": 80, "heat_k": 4000, "growth": 0.006},  
        "UK": {"lat": 51.50, "lon": -0.12, "base": 35000, "cool_k": 30, "heat_k": 2800, "growth": 0.005},
        "Italy": {"lat": 41.90, "lon": 12.49, "base": 40000, "cool_k": 800, "heat_k": 1500, "growth": 0.004},
        "Spain": {"lat": 40.41, "lon": -3.70, "base": 30000, "cool_k": 1000, "heat_k": 800, "growth": 0.008}
    },
    "North America": {
        "USA": {"lat": 39.00, "lon": -100.00, "base": 450000, "cool_k": 6000, "heat_k": 3000, "growth": 0.01}, 
        "Canada": {"lat": 43.65, "lon": -79.38, "base": 70000, "cool_k": 100, "heat_k": 5000, "growth": 0.012}, 
        "Mexico": {"lat": 23.63, "lon": -102.55, "base": 40000, "cool_k": 900, "heat_k": 100, "growth": 0.025}
    }
}

def calculate_thermodynamic_load(temp, base, cool_k, heat_k, year_diff, growth_rate):
    current_base = base * ((1 + growth_rate) ** year_diff)
    if temp > 22:
        return current_base + (cool_k * ((temp - 22) ** 1.3))
    elif temp < 15:
        return current_base + (heat_k * ((15 - temp) ** 1.2))
    else:
        return current_base

# حل مشكلة الكاش (استخدام اسم الدولة لضمان عدم اختلاط البيانات)
@st.cache_data(show_spinner=False, ttl=3600)
def generate_country_forecast(country_name, config):
    try:
        lat, lon = config['lat'], config['lon']
        base, cool_k, heat_k, growth = config['base'], config['cool_k'], config['heat_k'], config['growth']
        
        end_date = datetime.date.today() - datetime.timedelta(days=5)
        start_date = end_date - datetime.timedelta(days=365 * 6)
        
        # إضافة Timeout لمنع تهنيج السيرفر
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean&timezone=auto"
        response = requests.get(url, timeout=15)
        response.raise_for_status() # التأكد من نجاح الاتصال
        data = response.json()
        
        dates = data['daily']['time']
        temps = data['daily']['temperature_2m_mean']
        
        df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Temp_C': temps})
        df.dropna(inplace=True)
        
        min_date = df['Date'].min()
        df['Year_Diff'] = (df['Date'] - min_date).dt.days / 365.25
        df['Load_MW'] = df.apply(lambda row: calculate_thermodynamic_load(row['Temp_C'], base, cool_k, heat_k, row['Year_Diff'], growth), axis=1)
        df['Load_MW'] = df['Load_MW'] * np.random.uniform(0.98, 1.02, len(df))
        
        df_prophet = df[['Date', 'Load_MW']].rename(columns={'Date': 'ds', 'Load_MW': 'y'})
        
        # تدريب الموديل
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=6000)
        forecast = model.predict(future)
        return forecast
    except Exception as e:
        return None # في حالة فشل الاتصال

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
st.markdown("AI-Driven Capacity Planning utilizing thermodynamic principles.")

# Geographic Selection
st.sidebar.header("📍 Select Region")
selected_continent = st.sidebar.selectbox("Continent:", list(GLOBAL_GRID_CONFIG.keys()))
selected_country = st.sidebar.selectbox("Country:", list(GLOBAL_GRID_CONFIG[selected_continent].keys()))

country_config = GLOBAL_GRID_CONFIG[selected_continent][selected_country]

# Execute
with st.spinner(f"🛰️ Fetching Satellite Data & Training AI for {selected_country}... This may take a minute on first run."):
    forecast = generate_country_forecast(selected_country, country_config)

if forecast is None:
    st.error("🚨 Connection Error: Unable to fetch weather data from satellite API. Please try again in a few minutes.")
    st.stop()

# Date Selection
st.sidebar.header("📅 Select Target Date")
min_date = forecast['ds'].dt.date.values[-6000]
max_date = forecast['ds'].dt.date.values[-1]
selected_date = st.sidebar.date_input("Target Date (Up to 2040):", min_value=min_date, max_value=max_date, value=min_date + datetime.timedelta(days=365))

prediction_row = forecast[forecast['ds'].dt.date == selected_date]

if not prediction_row.empty:
    pred_load = prediction_row['yhat'].values[0]
    trend_val = prediction_row['trend'].values[0]

    st.markdown(f"### 🎯 Grid Load Projections for **{selected_country}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ Predicted Load", f"{pred_load:,.0f} MW", f"Based on {selected_country} profile")
    col2.metric("📊 Underlying Trend (Base Load)", f"{trend_val:,.0f} MW", "Long-term capacity need")
    col3.metric("🌡️ Grid Nature", "Summer-Peaking" if country_config['cool_k'] > country_config['heat_k'] else "Winter-Peaking")
    st.markdown("---")

    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        st.markdown(f"#### 📈 Long-Term Forecast (To 2040) - {selected_country}")
        forecast['Year'] = forecast['ds'].dt.year
        yearly_data = forecast.groupby('Year')['yhat'].mean().reset_index()
        fig_trend = px.line(yearly_data, x='Year', y='yhat', markers=True, line_shape='spline')
        fig_trend.update_traces(line_color='#007acc', line_width=3)
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### 🌦️ Seasonality (Summer vs Winter)")
        forecast['Month'] = forecast['ds'].dt.month_name()
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_data = forecast.groupby('Month')['yearly'].mean().reset_index()
        monthly_data['Month'] = pd.Categorical(monthly_data['Month'], categories=months_order, ordered=True)
        monthly_data = monthly_data.sort_values('Month')
        fig_season = px.bar(monthly_data, x='Month', y='yearly', color='yearly', color_continuous_scale='RdBu_r')
        fig_season.update_layout(showlegend=False)
        st.plotly_chart(fig_season, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>© 2026 PowerGuard AI by Eng. Mahmoud Reda. All rights reserved.</div>", unsafe_allow_html=True)