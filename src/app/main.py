import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="KomuterPulse",
    page_icon="🚆",
    layout="wide"
)

# --- Header Banner (Main Content) ---
st.markdown(
    """
    <div style='background: linear-gradient(90deg, #003366 0%, #0055a5 100%); padding: 1.5rem 2rem; border-radius: 18px; margin-bottom: 2rem;'>
        <h1 style='color: #fff; display: flex; align-items: center; gap: 1rem; margin: 0;'>
            <span style='font-size: 2.5rem;'>🚆</span> KomuterPulse
        </h1>
        <p style='color: #e0e0e0; font-size: 1.1rem; margin: 0.5rem 0 0 0;'>Real-time Transit Intelligence Platform for KTM Komuter</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
# Enhanced Sidebar Header and Layout
st.sidebar.markdown(
    """
    <style>
    .sidebar-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .sidebar-logo {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-title {
        color: #fff;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .sidebar-subtitle {
        color: #b0b0b0;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .sidebar-divider {
        border-top: 1px solid #444;
        margin: 1.2rem 0 1.2rem 0;
        width: 90%;
    }
    </style>
    <div class='sidebar-section'>
        <div class='sidebar-logo'>🚆</div>
        <div class='sidebar-title'>KomuterPulse</div>
        <div class='sidebar-subtitle'>Transit Intelligence Platform</div>
    </div>
    <div class='sidebar-divider'></div>
    """,
    unsafe_allow_html=True
)

# Styled Navigation Links
st.sidebar.markdown("### Navigation")
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        padding-top: 0 !important;
    }
    .sidebar .stButton button {
        width: 100%;
        text-align: left;
        padding-left: 1.5rem;
        border: none;
        background-color: transparent;
        color: inherit;
    }
    .sidebar .stButton button:hover {
        background-color: #f0f2f6;
        color: #003366;
    }
    .sidebar .stButton button:active {
        background-color: #e0e0e0;
        color: #003366;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- System Status Overview ---
st.markdown("""<hr style='margin-top:0; margin-bottom:1.5rem; border: 1px solid #e0e0e0;'>""", unsafe_allow_html=True)
st.subheader("System Status Overview")
col1, col2, col3, col4 = st.columns(4)

# Mock KPIs
total_riders = 125000
anomalies_today = 3
peak_hour = "07:00-08:00"
critical_routes = 5

with col1:
    st.markdown("""
    <div style='background: #f5f7fa; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 8px #e0e0e0;'>
        <h3 style='margin:0; color:#003366;'>👥</h3>
        <h2 style='margin:0; color:#0055a5;'>%s</h2>
        <p style='margin:0; color:#666;'>Total Riders (Today)</p>
    </div>""" % f"{total_riders:,}", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style='background: #f5f7fa; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 8px #e0e0e0;'>
        <h3 style='margin:0; color:#003366;'>🚨</h3>
        <h2 style='margin:0; color:#d7263d;'>%d</h2>
        <p style='margin:0; color:#666;'>Anomalies Detected</p>
    </div>""" % anomalies_today, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style='background: #f5f7fa; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 8px #e0e0e0;'>
        <h3 style='margin:0; color:#003366;'>⏰</h3>
        <h2 style='margin:0; color:#0055a5;'>%s</h2>
        <p style='margin:0; color:#666;'>Peak Hour</p>
    </div>""" % peak_hour, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div style='background: #f5f7fa; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 8px #e0e0e0;'>
        <h3 style='margin:0; color:#003366;'>🔥</h3>
        <h2 style='margin:0; color:#0055a5;'>%d</h2>
        <p style='margin:0; color:#666;'>Critical Routes</p>
    </div>""" % critical_routes, unsafe_allow_html=True)

st.markdown("""<hr style='margin-top:2rem; margin-bottom:1.5rem; border: 1px solid #e0e0e0;'>""", unsafe_allow_html=True)

# --- Real-time Alerts (Mock) ---
st.subheader("Real-time Alerts")
st.markdown("""
<div style='background: #fff3cd; border-left: 6px solid #ffecb5; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;'>
    <span style='font-size:1.3rem;'>⚠️</span>
    <b>Unusual spike detected on <span style='color:#d7263d;'>KL Sentral ↔️ Subang Jaya</span> (07:00-08:00).</b> Consider deploying additional train cars.
</div>
""", unsafe_allow_html=True)

st.markdown("""<hr style='margin-top:0; margin-bottom:1.5rem; border: 1px solid #e0e0e0;'>""", unsafe_allow_html=True)

# --- Key Metrics (Mock Data) ---
st.subheader("Key Metrics")
dates = pd.date_range(datetime.now() - timedelta(hours=23), periods=24, freq='H')
ridership = np.random.randint(4000, 7000, size=24)
metrics_df = pd.DataFrame({
    'Hour': dates.strftime('%H:%M'),
    'Ridership': ridership
})
fig = px.line(metrics_df, x='Hour', y='Ridership', title='Hourly Ridership (Mock Data)', markers=True)
fig.update_layout(xaxis_title='Hour', yaxis_title='Riders', height=350, plot_bgcolor='#f5f7fa', paper_bgcolor='#f5f7fa')
st.plotly_chart(fig, use_container_width=True)

st.markdown("""<hr style='margin-top:2rem; margin-bottom:1.5rem; border: 1px solid #e0e0e0;'>""", unsafe_allow_html=True)

# --- Feature Quick Access (Mock) ---
st.subheader("Quick Access to Features")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("🚦 Route Importance", use_container_width=True)
with col2:
    st.button("🕵️ Anomaly Detection", use_container_width=True)
with col3:
    st.button("🗓️ Schedule Recommendations", use_container_width=True)
with col4:
    st.button("🌱 Impact Analysis", use_container_width=True)

st.markdown("""<hr style='margin-top:2rem; margin-bottom:1.5rem; border: 1px solid #e0e0e0;'>""", unsafe_allow_html=True)

# --- Placeholder for further visualizations ---
st.subheader("Visualizations & Analytics")
st.info("Add more visualizations and analytics as you build out the platform.") 