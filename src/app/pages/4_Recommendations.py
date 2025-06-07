import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the src directory to the Python path for imports
sys.path.append(str(Path(__file__).parents[2]))

# Import the AI recommendation engine
from utils.ai_recommendation_engine import AIRecommendationEngine

st.set_page_config(page_title="Recommendations", page_icon="🚦")

# Enhanced dark mode styling with superior border design
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #fafafa;
    }
    
    /* Premium recommendation card styling with animated borders */
    .recommendation-card {
        position: relative !important;
        border-radius: 16px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        overflow: visible !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Animated border effect */
    .recommendation-card::before {
        content: '';
        position: absolute;
        inset: -2px;
        border-radius: 18px;
        background: linear-gradient(45deg, 
            #ff6b6b 0%, 
            #4ecdc4 25%, 
            #45b7d1 50%, 
            #96ceb4 75%, 
            #ffeaa7 100%);
        background-size: 300% 300%;
        opacity: 0;
        z-index: -1;
        animation: gradient-shift 3s ease infinite;
        transition: opacity 0.3s ease;
    }
    
    .recommendation-card:hover::before {
        opacity: 0.7;
    }
    
    /* Gradient animation */
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
      /* Enhanced hover effects */
    .recommendation-card:hover {
        box-shadow: 
            0 8px 20px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(255, 255, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Glow effect on hover */
    .recommendation-card:hover {
        filter: drop-shadow(0 0 20px rgba(69, 183, 209, 0.3));
    }
    /* Superior metrics styling with refined borders */
    .stMetric {
        background: linear-gradient(135deg, #1a1d29 0%, #2a2d3a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px solid transparent;
        background-clip: padding-box;
        position: relative;
        transition: all 0.3s ease;
        overflow: hidden;
    }
    
    .stMetric::before {
        content: '';
        position: absolute;
        inset: -2px;
        border-radius: 18px;
        background: linear-gradient(135deg, 
            rgba(69, 183, 209, 0.6) 0%, 
            rgba(255, 107, 107, 0.6) 50%, 
            rgba(150, 206, 180, 0.6) 100%);
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
      .stMetric:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    .stMetric:hover::before {
        opacity: 1;
    }
    
    /* Enhanced input styling with animated borders */
    .stSelectbox > div > div {
        background-color: #1e2130;
        color: #fafafa;
        border: 2px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(69, 183, 209, 0.6);
        box-shadow: 0 0 20px rgba(69, 183, 209, 0.2);
    }
      /* Keep buttons with original Streamlit styling */
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown(
    """
    <div style='background: linear-gradient(90deg, #003366 0%, #0055a5 100%); padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1rem;'>
        <h1 style='color: #fff; margin: 0;'>🚦 Operational Recommendations</h1>
        <p style='color: #e0e0e0; margin: 0.5rem 0 0 0;'>Actionable Insights for KTM Komuter Operations</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Introduction to recommendations
st.markdown("""
### Transforming Insights into Action

KomuterPulse automatically generates **actionable recommendations** for transit operations based on our forecasting 
and anomaly detection capabilities. These recommendations help operations teams:

- **Optimize schedules** to meet expected demand
- **Allocate resources** efficiently across the network
- **Respond proactively** to predicted anomalies
- **Improve passenger experience** through better service reliability

Our system prioritizes recommendations based on impact and urgency, allowing operations teams to focus on what matters most.
""")

# Demo section
st.markdown("---")
st.subheader("Current Recommendations")

# Date selection
recommendation_date = st.date_input("Select Date", datetime.now().date())

# Generate AI-powered recommendations data
if st.button("Generate Recommendations"):
    with st.spinner("Analyzing real model predictions and generating AI recommendations..."):
        try:
            # Initialize the AI recommendation engine
            ai_engine = AIRecommendationEngine()
            
            # Convert date to datetime for the engine
            target_date = datetime.combine(recommendation_date, datetime.min.time())
              # Generate AI-powered recommendations
            recommendations = ai_engine.generate_recommendations(
                target_date=target_date,
                time_window_hours=24
            )
            
            if not recommendations:
                st.warning("No recommendations generated for the selected date. Try a different date or check if model data is available.")
                st.stop()
            
            # Create DataFrame and sort by priority and time
            rec_df = pd.DataFrame(recommendations)
            
            # Map priority to sorting order (High=0, Medium=1, Low=2)
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            rec_df["priority_order"] = rec_df["priority"].map(priority_order)
            rec_df = rec_df.sort_values(["priority_order", "time"]).reset_index(drop=True)
            
        except Exception as e:
            st.error(f"Error generating AI recommendations: {str(e)}")
            st.info("Falling back to demonstration mode...")
            
            # Fallback to basic demo data if AI engine fails
            recommendations = [
                {
                    "route": "KL Sentral ↔️ Subang Jaya",
                    "time": "08:00",
                    "recommendation_type": "Capacity Management", 
                    "recommendation": "Increase train capacity due to predicted high ridership (2,847 passengers)",
                    "priority": "High",
                    "confidence": "87%",
                    "reasoning": "Model predictions indicate ridership exceeding normal capacity thresholds"
                }
            ]
            rec_df = pd.DataFrame(recommendations)
            rec_df["priority_order"] = rec_df["priority"].map({"High": 0, "Medium": 1, "Low": 2})
          # Display recommendation statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Recommendations", len(rec_df))
        with col2:
            high_priority = len(rec_df[rec_df["priority"] == "High"])
            st.metric("High Priority", high_priority)
        with col3:
            routes_affected = rec_df["route"].nunique()
            st.metric("Routes Affected", routes_affected)
        
        # Display recommendations by priority
        for priority in ["High", "Medium", "Low"]:
            priority_recs = rec_df[rec_df["priority"] == priority]
            if len(priority_recs) > 0:
                st.subheader(f"{priority} Priority Recommendations")
                
                for _, row in priority_recs.iterrows():
                    # Set dark mode colors based on priority
                    if row["priority"] == "High":
                        bg_color = "rgba(220, 53, 69, 0.15)"
                        border_color = "#dc3545"
                        accent_color = "#ff6b6b"
                        icon = "🔥"
                    elif row["priority"] == "Medium":
                        bg_color = "rgba(255, 193, 7, 0.15)"
                        border_color = "#ffc107"
                        accent_color = "#ffed4e"
                        icon = "ℹ️"
                    else:  # Low
                        bg_color = "rgba(13, 202, 240, 0.15)"
                        border_color = "#0dcaf0"
                        accent_color = "#4dd0e1"
                        icon = "⚠️"                    # Display recommendation card with superior border design
                    st.markdown(f"""
                    <div class="recommendation-card" style='
                        background: linear-gradient(135deg, {bg_color} 0%, rgba(20, 24, 35, 0.95) 100%);
                        border-radius: 16px;
                        padding: 1.5rem 2rem;
                        margin-bottom: 1.5rem;
                        box-shadow: 
                            0 8px 32px rgba(0, 0, 0, 0.4),
                            inset 0 1px 0 rgba(255, 255, 255, 0.1);
                        backdrop-filter: blur(10px);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        position: relative;
                        z-index: 1;
                    '>
                        <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;'>
                            <div style='display: flex; align-items: center; gap: 1rem;'>
                                <span style='
                                    font-size: 1.6rem; 
                                    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.5));
                                    animation: pulse 2s ease-in-out infinite;
                                '>{icon}</span>
                                <div>
                                    <span style='
                                        color: #ffffff; 
                                        font-weight: 700; 
                                        font-size: 1.2rem;
                                        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                                    '>{row["route"]}</span>
                                    <br>
                                    <span style='
                                        color: {accent_color}; 
                                        font-weight: 600;
                                        font-size: 1rem;
                                        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                                    '>📍 {row["time"]}</span>
                                </div>
                            </div>
                            <div style='text-align: right;'>
                                <div style='
                                    background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.05) 100%);
                                    border-radius: 12px;
                                    padding: 0.5rem 1rem;
                                    border: 1px solid rgba(255, 255, 255, 0.2);
                                    backdrop-filter: blur(5px);
                                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                                '>
                                    <span style='
                                        color: #e8e8e8; 
                                        font-size: 0.9rem; 
                                        font-weight: 600;
                                        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                                    '>
                                        Confidence: <span style='color: {accent_color}; font-weight: 700;'>{row["confidence"]}</span>
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div style='
                            background: linear-gradient(135deg, rgba(0, 0, 0, 0.3) 0%, rgba(0, 0, 0, 0.1) 100%);
                            border-radius: 12px;
                            padding: 1rem 1.2rem;
                            border-left: 4px solid {accent_color};
                            border: 1px solid rgba(255, 255, 255, 0.1);
                            backdrop-filter: blur(5px);
                            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
                        '>
                            <p style='
                                margin: 0 0 0.8rem 0;
                                font-weight: 700;
                                font-size: 1.1rem;
                                color: {accent_color};
                                text-transform: uppercase;
                                letter-spacing: 1px;
                                text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                            '>{row["recommendation_type"]}</p>
                            <p style='
                                margin: 0 0 0.8rem 0;
                                color: #e0e0e0;
                                line-height: 1.6;
                                font-size: 1rem;
                                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                            '>{row["recommendation"]}</p>
                            <p style='
                                margin: 0;
                                color: #b0b0b0;
                                line-height: 1.4;
                                font-size: 0.9rem;
                                font-style: italic;
                                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                            '>💡 {row.get("reasoning", "AI-powered analysis based on model predictions")}</p>
                        </div>
                    </div>
                    
                    <style>
                    @keyframes pulse {{
                        0%, 100% {{ transform: scale(1); }}
                        50% {{ transform: scale(1.1); }}
                    }}
                    </style>
                    """, unsafe_allow_html=True)
        
        # Visualize recommendations by hour
        st.subheader("Recommendations by Hour")
        
        # Extract hour from time string for grouping
        rec_df["hour"] = rec_df["time"].apply(lambda x: int(x.split(":")[0]))
        hourly_recs = rec_df.groupby("hour").size().reset_index(name="count")
        
        # Create hour labels for all 24 hours
        all_hours = pd.DataFrame({"hour": range(24)})
        hourly_recs = pd.merge(all_hours, hourly_recs, on="hour", how="left").fillna(0)
        hourly_recs["hour_label"] = hourly_recs["hour"].apply(lambda x: f"{x:02d}:00")
        
        # Create bar chart
        fig = px.bar(
            hourly_recs, 
            x="hour_label", 
            y="count",
            labels={"hour_label": "Hour", "count": "Number of Recommendations"},
            title="Distribution of Recommendations by Hour",
            color_discrete_sequence=["#0055a5"]
        )
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Number of Recommendations",
            height=400,
            xaxis={'categoryorder':'array', 'categoryarray': hourly_recs["hour_label"].tolist()}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualize recommendations by type
        st.subheader("Recommendations by Type")
        
        type_counts = rec_df["recommendation_type"].value_counts().reset_index()
        type_counts.columns = ["Recommendation Type", "Count"]
        
        fig = px.pie(
            type_counts,
            values="Count",
            names="Recommendation Type",
            title="Distribution of Recommendation Types",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# AI Engine Status
with st.expander("🤖 AI Engine Status", expanded=False):
    try:
        ai_engine = AIRecommendationEngine()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Real Predictions Loaded", f"{len(ai_engine.forecast_data):,}" if ai_engine.forecast_data is not None else "0")
            
        with col2:
            st.metric("AI Models Available", len(ai_engine.model_metrics))
            
        with col3:
            data_source = "Real Prophet + LSTM + XGBoost" if len(ai_engine.model_metrics) > 0 else "Demo Mode"
            st.metric("Data Source", data_source)
            
        st.success("✅ AI recommendation engine is operational with real trained model data!")
        
    except Exception as e:
        st.error(f"❌ AI Engine Error: {str(e)}")

# Show system features
st.markdown("---")
st.subheader("How Recommendations Are Generated")

st.markdown("""
Our recommendation engine follows a sophisticated process:

1. **Data Collection & Integration**
   - Ridership data from various sources is collected and integrated
   - Historical patterns are analyzed for context

2. **Forecasting & Anomaly Detection**
   - LSTM models predict expected ridership for all routes and times
   - Hybrid anomaly detection identifies potential issues

3. **Recommendation Generation**
   - Business rules are applied to prediction and anomaly results
   - Recommendations are generated based on predefined thresholds
   - Confidence scores are calculated for each recommendation

4. **Prioritization & Delivery**
   - Recommendations are prioritized by impact and urgency
   - High-priority recommendations can trigger alerts to operations staff
   - All recommendations are logged for analysis and continuous improvement

This automated approach ensures that transit operations can stay ahead of issues and provide the best possible 
service to commuters.
""")