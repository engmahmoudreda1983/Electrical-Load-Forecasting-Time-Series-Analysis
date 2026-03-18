# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
# Proprietary and Confidential.
# Unauthorized copying, modification, distribution, or reverse engineering 
# of this software, including the physics-informed AI guardrails, is strictly prohibited.
# ==============================================================================

import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
import plotly.graph_objs as go
import datetime

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="PowerGuard AI - Load Forecasting", page_icon="⚡", layout="wide")

# ==========================================
# --- 2. نظام تسجيل الدخول (Login System) ---
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #4da6ff;'>⚡ PowerGuard AI - Electrical Load Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Restricted Access</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 1px solid #333;'>", unsafe_allow_html=True)
        # --- قم بتعديل اليوزر والباسورد هنا ليتطابق مع مشروعك السابق ---
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login", use_container_width=True)
        
        if login_btn:
            if username == "admin" and password == "12345": 
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("❌ Invalid Username or Password")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
    st.stop() # إيقاف تشغيل باقي الكود إذا لم يسجل الدخول

# ==========================================
# --- 3. التطبيق الرئيسي (Main Dashboard) ---
# ==========================================

# Sidebar Logout
st.sidebar.markdown("### 👤 User Profile")
st.sidebar.markdown("**Eng. Mahmoud Reda**")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state['logged_in'] = False
    st.rerun()
st.sidebar.markdown("---")

st.title("⚡ PowerGuard AI - Electrical Load Forecasting")
st.markdown("Strategic Predictive Analytics for Power Grid Operations (1-Year Horizon).")

# Load the Pre-trained Model
@st.cache_resource
def load_model():
    with open('prophet_model.json', 'r') as fin:
        model = model_from_json(fin.read())
    return model

model = load_model()

# Generate Forecast
@st.cache_data
def get_forecast():
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast

forecast = get_forecast()

# Sidebar for Interactivity
st.sidebar.header("📅 Select Forecast Date")
future_dates = forecast['ds'].dt.date.tail(365).values

selected_date = st.sidebar.date_input(
    "Choose a future date to predict load:", 
    min_value=future_dates[0], 
    max_value=future_dates[-1], 
    value=future_dates[0]
)

# Filter Results
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

    # Interactive Plotly Chart
    st.markdown("### 📈 Load Trend Analysis (Forecast)")
    fig = go.Figure()
    
    # Plot 365 days of predicted data
    future_forecast = forecast.tail(365)
    fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Predicted Load', line=dict(color='blue', width=2)))
    
    # Highlight selected date
    fig.add_trace(go.Scatter(x=[selected_date], y=[pred_load], mode='markers', name='Selected Target', marker=dict(color='red', size=14, symbol='star')))

    fig.update_layout(xaxis_title="Date", yaxis_title="Electrical Load (MW)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# --- 4. حقوق الملكية (Footer) ---
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.<br>
    Proprietary and Confidential.
</div>
""", unsafe_allow_html=True)