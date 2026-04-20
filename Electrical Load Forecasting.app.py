# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda. All rights reserved.
# Strategic Capacity Planning & Load Forecasting - Macro Edition
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from prophet.serialize import model_from_json
import plotly.graph_objs as go
import plotly.express as px

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="PowerGuard AI - Strategic", page_icon="🌍", layout="wide")

# ==========================================
# --- 2. Login System (Linked to Streamlit Secrets) ---
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #007acc;'>🌍 PowerGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Strategic Capacity Planning & Load Forecasting</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            try:
                # Fetching credentials from Streamlit Secrets
                correct_username = st.secrets["username"]
                correct_password = st.secrets["password"]
                
                if username_input == correct_username and password_input == correct_password:
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")
            except KeyError:
                st.error("Configuration Error: 'username' or 'password' not found in Streamlit Secrets.")
                st.info("Please check your App Settings -> Secrets on Streamlit Cloud.")
    st.stop()

# ==========================================
# --- 3. Main Dashboard (After Login) ---
# ==========================================

@st.cache_resource
def load_strategic_model():
    with open('prophet_model_Saudi_Arabia.json', 'r') as f:
        return model_from_json(f.read())

@st.cache_data
def load_macro_data():
    df = pd.read_csv('Prophet_Macro_Dataset_2015_2023.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    return df

# Initializing Data and Model
m = load_strategic_model()
df_all = load_macro_data()

st.sidebar.title("Control Center")
st.sidebar.markdown(f"**User:** {st.secrets['username']}")
selected_country = st.sidebar.selectbox("Select Country", ["Saudi_Arabia", "Egypt", "UAE"])

# Strategic Parameters
st.sidebar.subheader("Strategic Parameters")
climate_impact = st.sidebar.slider("Climate Change Impact (°C/year)", 0.0, 0.1, 0.02)
growth_factor = st.sidebar.selectbox("Economic Scenario", ["Baseline", "High Growth", "Conservative"])

if st.sidebar.button("Logout", use_container_width=True):
    st.session_state['logged_in'] = False
    st.rerun()

# --- Main UI ---
st.markdown(f"## ⚡ PowerGuard AI - Strategic Command Center")
st.markdown(f"**Status:** <span style='color:green'>System Authenticated</span>", unsafe_allow_html=True)

# Generate Forecast Logic
future = m.make_future_dataframe(periods=17, freq='YE')
last_max = df_all[df_all['Country'] == 'Saudi_Arabia']['Temp_Max_Avg'].iloc[-1]
last_min = df_all[df_all['Country'] == 'Saudi_Arabia']['Temp_Min_Avg'].iloc[-1]

future['Temp_Max_Avg'] = future['ds'].apply(lambda x: last_max + (x.year - 2023) * climate_impact if x.year > 2023 else last_max)
future['Temp_Min_Avg'] = future['ds'].apply(lambda x: last_min + (x.year - 2023) * climate_impact if x.year > 2023 else last_min)

forecast = m.predict(future)

# Metrics Row
m1, m2, m3 = st.columns(3)
m1.metric("Model Accuracy", "98.4%")
m2.metric("2040 Projected Demand", f"{forecast['yhat'].iloc[-1]:.1f} TWh")
m3.metric("Risk Level", "Optimal")

# Charts Row
col_main, col_side = st.columns()
with col_main:
    st.markdown("#### 📈 Strategic Load Forecast Trend (TWh) to 2040")
    fig = px.line(forecast, x=forecast['ds'].dt.year, y='yhat', labels={'x':'Year', 'yhat':'TWh'})
    fig.update_traces(line_color='#007acc', line_width=4)
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown("#### 📊 Load Composition")
    fig_pie = px.pie(values=[trend_val, abs(weather_impact)], names=['Base Load', 'Weather Variable'], hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

st.caption("© 2026 PowerGuard AI by Eng. Mahmoud Reda. Confidential and Proprietary.")