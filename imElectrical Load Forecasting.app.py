# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
# Proprietary and Confidential.
# ==============================================================================

import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
import plotly.graph_objs as go
import plotly.express as px
import datetime

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="PowerGuard AI - Load Forecasting", page_icon="⚡", layout="wide")

# ==========================================
# --- 2. نظام تسجيل الدخول (Login System) ---
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #007acc;'>⚡ PowerGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Strategic Electrical Load Forecasting</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        # تمت إزالة المربع الأسود ليصبح التصميم أنظف
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Secure Login", use_container_width=True)
        
        if login_btn:
            if username == "admin" and password == "DBA2026": 
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("❌ Invalid Username or Password")
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        © 2026 PowerGuard AI by Eng. Mahmoud Reda. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ==========================================
# --- 3. التطبيق الرئيسي (Main Dashboard) ---
# ==========================================

# Sidebar
st.sidebar.markdown("### 👤 User Profile")
st.sidebar.markdown("**Eng. Mahmoud Reda**\n\n*DBA Candidate - Strategic Management*")
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state['logged_in'] = False
    st.rerun()
st.sidebar.markdown("---")

st.title("⚡ PowerGuard AI - Long-Term Capacity Planning")
st.markdown("Strategic Predictive Analytics for Power Grid Operations (Forecast Horizon up to **2040**).")

# Load Model
@st.cache_resource
def load_model():
    with open('prophet_model.json', 'r') as fin:
        model = model_from_json(fin.read())
    return model

model = load_model()

# Generate Future Data up to 2040 (approx 8500 days from the end of training data)
@st.cache_data
def get_forecast():
    future = model.make_future_dataframe(periods=8500)
    forecast = model.predict(future)
    return forecast

with st.spinner("Generating Long-Term Forecast Data up to 2040... Please Wait."):
    forecast = get_forecast()

# Extract Dates
future_dates = forecast['ds'].dt.date.values
min_date = future_dates[0]
max_date = future_dates[-1]

# Set Default date to something in the future like 2026
default_date = datetime.date(2026, 6, 15)
if default_date > max_date or default_date < min_date:
    default_date = max_date

# Sidebar Date Picker
st.sidebar.header("📅 Select Target Date")
selected_date = st.sidebar.date_input(
    "Choose a date (up to 2040):", 
    min_value=min_date, 
    max_value=max_date, 
    value=default_date
)

# Filter Data for Selected Date
prediction_row = forecast[forecast['ds'].dt.date == selected_date]

if not prediction_row.empty:
    pred_load = prediction_row['yhat'].values[0]
    pred_min = prediction_row['yhat_lower'].values[0]
    pred_max = prediction_row['yhat_upper'].values[0]
    trend_val = prediction_row['trend'].values[0]

    # --- Section 1: Top KPI Metrics ---
    st.markdown("### 🎯 Grid Load Projections")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚡ Predicted Load", f"{pred_load:,.0f} MW", "Target")
    col2.metric("📉 Base Expected (Min)", f"{pred_min:,.0f} MW")
    col3.metric("📈 Peak Expected (Max)", f"{pred_max:,.0f} MW")
    col4.metric("📊 Underlying Trend", f"{trend_val:,.0f} MW", "Capacity Growth")
    st.markdown("---")

    # --- Section 2: Macro-Level Strategic Charts ---
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.markdown("#### 📈 Long-Term Annual Load Trend (To 2040)")
        forecast['Year'] = forecast['ds'].dt.year
        yearly_data = forecast.groupby('Year')['yhat'].mean().reset_index()
        
        fig_trend = px.line(yearly_data, x='Year', y='yhat', markers=True, line_shape='spline')
        fig_trend.update_traces(line_color='#007acc', line_width=3, marker=dict(size=6, color='red'))
        fig_trend.update_layout(xaxis_title="Year", yaxis_title="Average Annual Load (MW)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### 🌦️ Monthly Seasonal Impact")
        forecast['Month'] = forecast['ds'].dt.month_name()
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_data = forecast.groupby('Month')['yearly'].mean().reset_index()
        monthly_data['Month'] = pd.Categorical(monthly_data['Month'], categories=months_order, ordered=True)
        monthly_data = monthly_data.sort_values('Month')
        
        fig_season = px.bar(monthly_data, x='Month', y='yearly', color='yearly', color_continuous_scale='RdBu_r')
        fig_season.update_layout(xaxis_title="", yaxis_title="Variance from Avg (MW)", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_season, use_container_width=True)

    # --- Section 3: Micro-Level Daily Chart ---
    st.markdown(f"#### 🔍 Micro-View: 30-Day Window around {selected_date}")
    window_start = selected_date - datetime.timedelta(days=15)
    window_end = selected_date + datetime.timedelta(days=15)
    window_data = forecast[(forecast['ds'].dt.date >= window_start) & (forecast['ds'].dt.date <= window_end)]
    
    fig_micro = go.Figure()
    fig_micro.add_trace(go.Scatter(x=window_data['ds'], y=window_data['yhat'], mode='lines+markers', name='Predicted Load', line=dict(color='#007acc', width=2)))
    fig_micro.add_trace(go.Scatter(x=window_data['ds'], y=window_data['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='lightgray', dash='dot')))
    fig_micro.add_trace(go.Scatter(x=window_data['ds'], y=window_data['yhat_lower'], mode='lines', name='Lower Bound', fill='tonexty', fillcolor='rgba(200, 200, 200, 0.2)', line=dict(color='lightgray', dash='dot')))
    fig_micro.add_trace(go.Scatter(x=[pd.to_datetime(selected_date)], y=[pred_load], mode='markers', name='Selected Date', marker=dict(color='red', size=14, symbol='star')))
    
    fig_micro.update_layout(xaxis_title="Date", yaxis_title="Electrical Load (MW)", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_micro, use_container_width=True)

# ==========================================
# --- 4. حقوق الملكية (Footer) ---
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    © 2026 PowerGuard AI by Eng. Mahmoud Reda. All rights reserved.<br>
    Proprietary and Confidential.
</div>
""", unsafe_allow_html=True)