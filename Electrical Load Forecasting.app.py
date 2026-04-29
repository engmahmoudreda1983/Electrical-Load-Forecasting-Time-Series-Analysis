# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda. All rights reserved.
# Proprietary and Confidential.
# ==============================================================================

import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
import plotly.graph_objs as go
import plotly.express as px
import datetime
import numpy as np

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="PowerGuard AI - Global Load", page_icon="🌍", layout="wide")

# ==========================================
# --- 2. Login System (Bulletproof Secrets) ---
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #007acc;'>🌍 PowerGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Global Capacity Planning & Load Forecasting</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        if st.button("Secure Login", use_container_width=True):
            
            # Fetching credentials safely
            valid_user = st.secrets.get("username", "admin")
            valid_pass = st.secrets.get("password", "dba2026")
            
            if username_input == valid_user and password_input == valid_pass: 
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("❌ Invalid Username or Password")
                
            if "username" not in st.secrets:
                st.toast("⚠️ تنبيه: النظام يعمل بالبيانات الافتراضية لأن السيرفر لم يقرأ الأسرار.")
                
    st.stop()

# ==========================================
# --- 3. Global Engineering Database ---
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

@st.cache_data(show_spinner=False, ttl=3600)
def generate_country_forecast(country_name, config):
    try:
        # قراءة الموديل الجديد أوفلاين
        with open('prophet_model_Saudi_Arabia.json', 'r') as f:
            model = model_from_json(f.read())
        
        # إنشاء بيانات يومية حتى 2040
        future = pd.DataFrame({'ds': pd.date_range(start='2015-01-01', end='2040-12-31', freq='D')})
        
        future['Temp_Max_Avg'] = future['ds'].apply(lambda x: 33.1 + (x.year - 2023)*0.02 if x.year > 2023 else 33.1)
        future['Temp_Min_Avg'] = future['ds'].apply(lambda x: 19.1 + (x.year - 2023)*0.02 if x.year > 2023 else 19.1)
        
        forecast = model.predict(future)
        
        scale_factor = (1000000 / 8760) 
        forecast['yhat'] = forecast['yhat'] * scale_factor
        forecast['trend'] = forecast['trend'] * scale_factor
        
        is_summer = (forecast['ds'].dt.month >= 5) & (forecast['ds'].dt.month <= 9)
        forecast['yearly'] = np.where(
            is_summer,
            forecast['trend'] * 0.15, 
            -forecast['trend'] * 0.10  
        )
        forecast['yhat'] = forecast['trend'] + forecast['yearly']
        
        return forecast
    except Exception as e:
        return None 

# ==========================================
# --- 4. Main Dashboard UI ---
# ==========================================
st.sidebar.markdown("### 👤 User Profile")
st.sidebar.markdown("**Eng. Mahmoud Reda**\n\n*DBA Candidate*")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state['logged_in'] = False
    st.rerun()
st.sidebar.markdown("---")

st.title("🌍 Global Electrical Load Forecasting")
st.markdown("AI-Driven Capacity Planning utilizing thermodynamic principles.")

st.sidebar.header("📍 Select Region")
selected_continent = st.sidebar.selectbox("Continent:", list(GLOBAL_GRID_CONFIG.keys()))
selected_country = st.sidebar.selectbox("Country:", list(GLOBAL_GRID_CONFIG[selected_continent].keys()))

country_config = GLOBAL_GRID_CONFIG[selected_continent][selected_country]

with st.spinner(f"🛰️ Loading AI Model for {selected_country}..."):
    forecast = generate_country_forecast(selected_country, country_config)

if forecast is None:
    st.error("🚨 Error: Unable to load offline model. Ensure JSON file is in directory.")
    st.stop()

st.sidebar.header("📅 Select Target Date")
min_date = forecast['ds'].min().date()
max_date = forecast['ds'].max().date()
selected_date = st.sidebar.date_input("Target Date (Up to 2040):", min_value=min_date, max_value=max_date, value=datetime.date(2026, 7, 15))

# تحويل التواريخ لنصوص لضمان التطابق 100%
selected_date_str = selected_date.strftime('%Y-%m-%d')
forecast['date_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')
prediction_row = forecast[forecast['date_str'] == selected_date_str]

if not prediction_row.empty:
    # <--- التعديلات هنا لضمان استخراج رقم واحد فقط بدل قائمة
    pred_load = float(prediction_row['yhat'].iloc[0])
    trend_val = float(prediction_row['trend'].iloc[0])
    
    weather_impact = pred_load - trend_val
    impact_pct = (abs(weather_impact) / pred_load) * 100 if pred_load > 0 else 0

    st.markdown(f"### 🎯 Grid Load Projections for **{selected_country}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ Predicted Load", f"{pred_load:,.0f} MW", f"Based on {selected_country} profile")
    col2.metric("📊 Underlying Trend (Base Load)", f"{trend_val:,.0f} MW", "Long-term capacity need")
    col3.metric("🌡️ Grid Nature", "Summer-Peaking" if country_config['cool_k'] > country_config['heat_k'] else "Winter-Peaking")
    
    st.markdown("---")
    
    st.markdown("### 🧠 Executive Decision Insight")
    col_text, col_pie = st.columns([1.5, 1])
    
    with col_text:
        st.markdown("<br>", unsafe_allow_html=True)
        
        if weather_impact > 1000:
            st.warning(f"⚠️ **Peak Demand Alert:** \nThe weather conditions (Heating/Cooling) will add an extra **{weather_impact:,.0f} MW** ({impact_pct:.1f}% of total demand) to the base load on {selected_date}.")
            st.markdown("**💡 Strategic Action:** Ensure Peaking Power Plants (e.g., Gas Turbines) or Battery Energy Storage Systems (BESS) are scheduled and available to cover this surge.")
        elif weather_impact < -1000:
            st.info(f"📉 **Low Demand Period:** \nThe expected load is below the base trend by **{abs(weather_impact):,.0f} MW** due to highly favorable weather conditions.")
            st.markdown("**💡 Strategic Action:** This represents an optimal window for scheduling preventative maintenance for major Base-load power plants without risking supply.")
        else:
            st.success(f"✅ **Stable Operation:** \nThe expected load is almost identical to the base trend with minimal weather interference (Variance: **{weather_impact:,.0f} MW**).")
            st.markdown("**💡 Strategic Action:** Proceed with standard grid operation protocols. No extreme interventions required.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 💸 Financial & Operational Impact")
        
        daily_cost_variance = weather_impact * 24 * 50 
        max_capacity = trend_val * 1.3
        stress_level = (pred_load / max_capacity) * 100 if max_capacity > 0 else 0
        stress_level = min(max(stress_level, 0), 100) 
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Est. Daily OPEX Variance ($)", 
                   f"${abs(daily_cost_variance):,.0f}", 
                   f"{'Cost Overrun' if weather_impact > 0 else 'Cost Savings'}", 
                   delta_color="inverse" if weather_impact > 0 else "normal")
        
        with mc2:
            st.write(f"**⚡ Grid Stress Level:** {stress_level:.1f}%")
            if stress_level > 85:
                st.progress(stress_level / 100.0)
                st.caption("🔴 High Stress (Risk of brownouts)")
            elif stress_level > 70:
                st.progress(stress_level / 100.0)
                st.caption("🟡 Moderate Stress")
            else:
                st.progress(stress_level / 100.0)
                st.caption("🟢 Optimal Load")

    with col_pie:
        fig_pie = px.pie(values=[trend_val, abs(weather_impact)], names=['Base Load', 'Weather Variable'], hole=0.4)
        fig_pie.update_layout(title_text="Load Composition", title_x=0.5, margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    col_chart1, col_chart2 = st.columns() # <--- التعديل هنا كمان!
    
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
        
else:
    st.error(f"🚨 عذراً، لا توجد بيانات مسجلة في الموديل لتاريخ {selected_date}.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>© 2026 PowerGuard AI by Eng. Mahmoud Reda. All rights reserved.</div>", unsafe_allow_html=True)