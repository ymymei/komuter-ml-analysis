import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from src.app.utils.model_loader import ModelManager
from src.app.utils.navigation import render_custom_navigation, add_sidebar_logo

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Dashboard - KomuterPulse",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@komuterpulse.my',
        'Report a bug': 'mailto:bugs@komuterpulse.my',
        'About': 'KomuterPulse v1.0.0 - Real-time Transit Intelligence Platform'
    }
)

# Initialize model manager in session state if not already present
if 'model_manager' not in st.session_state:
    # Create a placeholder for loading messages
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        with st.spinner("Loading ML models..."):
            st.session_state.model_manager = ModelManager()
            load_success = st.session_state.model_manager.load_models()
            if not load_success:
                st.warning("Some models failed to load. Some functionality may be limited.")
    
    # Clear the loading messages after models are loaded
    loading_placeholder.empty()

# Set dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .st-emotion-cache-bqe0wn {
        background-color: #0e1117;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #fafafa;
    }
    .metric-container {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .feature-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .alert-card-high {
        background: rgba(220, 53, 69, 0.2);
        border-left: 6px solid #dc3545;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .alert-card-medium {
        background: rgba(255, 193, 7, 0.2);
        border-left: 6px solid #ffc107;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .alert-card-low {
        background: rgba(13, 202, 240, 0.2);
        border-left: 6px solid #0dcaf0;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .header-banner {
        background: linear-gradient(90deg, #0d1b2a 0%, #1b263b 100%);
        padding: 1.5rem 2rem;
        border-radius: 18px;
        margin-bottom: 2rem;
    }
    .plotly-graph {
        background-color: transparent !important;
    }
    .plot-container {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Banner (Main Content) ---
st.markdown(
    """
    <div class="header-banner">
        <h1 style='color: #fff; display: flex; align-items: center; gap: 1rem; margin: 0;'>
            <span style='font-size: 2.5rem;'>🚆</span> KomuterPulse
        </h1>
        <p style='color: #e0e0e0; font-size: 1.1rem; margin: 0.5rem 0 0 0;'>Real-time Transit Intelligence Platform for KTM Komuter</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Project Description
st.markdown("""
## About KomuterPulse
KomuterPulse is a comprehensive real-time transit intelligence platform that transforms raw KTM Komuter ridership data into actionable insights. 
Our solution combines advanced time series forecasting with anomaly detection using a hybrid AI approach to revolutionize transit management.

### Key Features:
- **Time Series Forecasting**: LSTM neural networks for accurate ridership prediction
- **Anomaly Detection**: Hybrid approach combining LSTM residuals with Isolation Forest
- **Actionable Recommendations**: Automated operations suggestions based on predicted patterns
- **Multi-step Forecasting**: Look ahead capability for proactive management
""")

# --- Sidebar ---
# The built-in Streamlit navigation will appear here automatically

# --- System Status Overview ---
st.markdown("""<hr style='margin-top:0; margin-bottom:1.5rem; border: 1px solid #2a3042;'>""", unsafe_allow_html=True)
st.subheader("System Status Overview")
col1, col2, col3, col4 = st.columns(4)

# Mock KPIs
total_riders = 125000
anomalies_today = 3
peak_hour = "07:00-08:00"
critical_routes = 5

with col1:
    st.markdown("""
    <div class="metric-container">
        <h3 style='margin:0; color:#adb5bd;'>👥</h3>
        <h2 style='margin:0; color:#4895ef;'>%s</h2>
        <p style='margin:0; color:#adb5bd;'>Total Riders (Today)</p>
    </div>""" % f"{total_riders:,}", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-container">
        <h3 style='margin:0; color:#adb5bd;'>🚨</h3>
        <h2 style='margin:0; color:#f94144;'>%d</h2>
        <p style='margin:0; color:#adb5bd;'>Anomalies Detected</p>
    </div>""" % anomalies_today, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-container">
        <h3 style='margin:0; color:#adb5bd;'>⏰</h3>
        <h2 style='margin:0; color:#90be6d;'>%s</h2>
        <p style='margin:0; color:#adb5bd;'>Peak Hour</p>
    </div>""" % peak_hour, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-container">
        <h3 style='margin:0; color:#adb5bd;'>🔥</h3>
        <h2 style='margin:0; color:#f9c74f;'>%d</h2>
        <p style='margin:0; color:#adb5bd;'>Critical Routes</p>
    </div>""" % critical_routes, unsafe_allow_html=True)

st.markdown("""<hr style='margin-top:2rem; margin-bottom:1.5rem; border: 1px solid #2a3042;'>""", unsafe_allow_html=True)

# --- Real-time Alerts (Mock) ---
st.subheader("Real-time Anomaly Alerts")

alerts = [
    {"route": "KL Sentral ↔️ Subang Jaya", "time": "07:00-08:00", "severity": "High", 
     "message": "Unusual spike detected. Consider deploying additional train cars."},
    {"route": "Bandar Tasek Selatan ↔️ Kajang", "time": "17:30-18:30", "severity": "Medium", 
     "message": "Ridership above 85% capacity. Monitor situation."},
    {"route": "Kepong ↔️ Segambut", "time": "12:00-13:00", "severity": "Low", 
     "message": "Minor deviation from expected patterns."}
]

for i, alert in enumerate(alerts):
    if alert["severity"] == "High":
        card_class = "alert-card-high"
        icon = "🔥"
    elif alert["severity"] == "Medium":
        card_class = "alert-card-medium"
        icon = "ℹ️"
    else:
        card_class = "alert-card-low"
        icon = "⚠️"
    
    st.markdown(f"""
    <div class="{card_class}">
        <span style='font-size:1.3rem;'>{icon}</span>
        <b>{alert["route"]}</b> ({alert["time"]}) - <span style='color:#f94144;'>{alert["severity"]} Priority</span>
        <p style='margin-top: 0.5rem; margin-bottom: 0;'>{alert["message"]}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""<hr style='margin-top:1rem; margin-bottom:1.5rem; border: 1px solid #2a3042;'>""", unsafe_allow_html=True)

# --- Key Metrics - Hourly Ridership with Anomaly Detection ---
st.subheader("Hourly Ridership with Anomaly Detection")

# Generate mock data for visualization
dates = pd.date_range(datetime.now() - timedelta(hours=23), periods=24, freq='h')
ridership = np.random.randint(4000, 7000, size=24)
# Simulate some anomalies
ridership[5] = 8500  # Morning anomaly
ridership[17] = 8200  # Evening anomaly

# Add predictions that are close to actual but not exact
predictions = ridership + np.random.normal(0, 300, size=24)
predictions[5] = 6500  # Show the model didn't predict this anomaly
predictions[17] = 6800  # Show the model didn't predict this anomaly

# Add confidence intervals
upper_bound = predictions + 1000
lower_bound = predictions - 1000

# Create DataFrame
metrics_df = pd.DataFrame({
    'Hour': dates.strftime('%H:%M'),
    'Actual': ridership,
    'Predicted': predictions,
    'Upper Bound': upper_bound,
    'Lower Bound': lower_bound,
    'Anomaly': (ridership > upper_bound) | (ridership < lower_bound)
})

# Create a more sophisticated plot with anomalies highlighted
fig = go.Figure()

# Add confidence interval as a shaded area
fig.add_trace(go.Scatter(
    x=metrics_df['Hour'],
    y=metrics_df['Upper Bound'],
    fill=None,
    mode='lines',
    line_color='rgba(0,100,80,0)',
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=metrics_df['Hour'],
    y=metrics_df['Lower Bound'],
    fill='tonexty',
    mode='lines',
    line_color='rgba(0,100,80,0)',
    fillcolor='rgba(73, 149, 239, 0.2)',
    name='Confidence Interval'
))

# Add the predicted line
fig.add_trace(go.Scatter(
    x=metrics_df['Hour'],
    y=metrics_df['Predicted'],
    mode='lines+markers',
    line=dict(color='#4895ef', width=2),
    name='Predicted Ridership'
))

# Add the actual line
fig.add_trace(go.Scatter(
    x=metrics_df['Hour'],
    y=metrics_df['Actual'],
    mode='lines+markers',
    line=dict(color='#90be6d', width=2),
    name='Actual Ridership'
))

# Highlight anomalies
anomaly_hours = metrics_df[metrics_df['Anomaly']]['Hour']
anomaly_values = metrics_df[metrics_df['Anomaly']]['Actual']

fig.add_trace(go.Scatter(
    x=anomaly_hours,
    y=anomaly_values,
    mode='markers',
    marker=dict(color='#f94144', size=12, symbol='circle-open', line=dict(width=2)),
    name='Anomalies'
))

# Update layout
fig.update_layout(
    title='Hourly Ridership with Anomaly Detection',
    xaxis_title='Hour',
    yaxis_title='Ridership',
    height=400,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color="#fafafa"),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
)

# Add the graph with transparent background
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("""<hr style='margin-top:1rem; margin-bottom:1.5rem; border: 1px solid #2a3042;'>""", unsafe_allow_html=True)

# --- Feature Quick Access ---
st.subheader("Quick Access to Features")
st.markdown("Navigate to specific analytics modules to explore detailed insights:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <a href="/Forecasting" target="_self" style="text-decoration: none">
        <div class="feature-card">
            <h3 style="margin:0; color:#adb5bd;">📈</h3>
            <h4 style="margin:0.5rem 0; color:#4895ef;">Ridership Forecasting</h4>
            <p style="margin:0; color:#adb5bd; font-size: 0.9rem;">Time series predictions with LSTM models</p>
        </div>
    </a>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <a href="/Anomaly_Detection" target="_self" style="text-decoration: none">
        <div class="feature-card">
            <h3 style="margin:0; color:#adb5bd;">🔍</h3>
            <h4 style="margin:0.5rem 0; color:#4895ef;">Anomaly Detection</h4>
            <p style="margin:0; color:#adb5bd; font-size: 0.9rem;">Hybrid approach to identify unusual patterns</p>
        </div>
    </a>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <a href="/Trip_Analysis" target="_self" style="text-decoration: none">
        <div class="feature-card">
            <h3 style="margin:0; color:#adb5bd;">📍</h3>
            <h4 style="margin:0.5rem 0; color:#4895ef;">Trip Analysis</h4>
            <p style="margin:0; color:#adb5bd; font-size: 0.9rem;">Route-specific insights and predictions</p>
        </div>
    </a>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <a href="/Recommendations" target="_self" style="text-decoration: none">
        <div class="feature-card">
            <h3 style="margin:0; color:#adb5bd;">🚦</h3>
            <h4 style="margin:0.5rem 0; color:#4895ef;">Recommendations</h4>
            <p style="margin:0; color:#adb5bd; font-size: 0.9rem;">Actionable insights for operations</p>
        </div>
    </a>
    """, unsafe_allow_html=True) 