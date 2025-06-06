import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Recommendations", page_icon="🚦")

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

# Generate mock recommendations data
if st.button("Generate Recommendations"):
    with st.spinner("Analyzing data and generating recommendations..."):
        # Create mock data - routes with various recommendations
        routes = [
            "KL Sentral ↔️ Subang Jaya",
            "Bandar Tasek Selatan ↔️ Kajang",
            "KL Sentral ↔️ Kepong",
            "Subang Jaya ↔️ Batu Caves",
            "KL Sentral ↔️ Shah Alam",
            "Kepong ↔️ Sungai Buloh"
        ]
        
        # Morning peak hours
        morning_hours = ["07:00", "07:30", "08:00", "08:30"]
        
        # Evening peak hours
        evening_hours = ["17:00", "17:30", "18:00", "18:30", "19:00"]
        
        # Recommendation types
        rec_types = [
            "Add train capacity",
            "Increase frequency",
            "Schedule adjustment",
            "Monitor crowding",
            "Address potential delay"
        ]
        
        # Severities
        severities = ["High", "Medium", "Low"]
        severity_weights = [0.5, 0.3, 0.2]  # Probability weights
        
        # Generate random recommendations
        recommendations = []
        num_recommendations = np.random.randint(8, 15)
        
        for _ in range(num_recommendations):
            route = np.random.choice(routes)
            
            # More recommendations during peak hours
            if np.random.random() < 0.7:  # 70% chance of peak hour recommendation
                if np.random.random() < 0.5:  # 50/50 split between morning and evening peak
                    time = np.random.choice(morning_hours)
                    rec_type = np.random.choice(rec_types[:3])  # More likely capacity/frequency during peaks
                else:
                    time = np.random.choice(evening_hours)
                    rec_type = np.random.choice(rec_types[:3])
            else:
                # Off-peak hours
                hour = np.random.randint(9, 17)
                time = f"{hour:02d}:00"
                rec_type = np.random.choice(rec_types[2:])  # More likely monitoring/adjustments
            
            severity = np.random.choice(severities, p=severity_weights)
            
            # Generate specific recommendation text based on type
            if rec_type == "Add train capacity":
                detail = f"Add {np.random.randint(1, 3)} additional cars due to predicted high ridership."
            elif rec_type == "Increase frequency":
                detail = f"Reduce headway to {np.random.randint(5, 15)} minutes to accommodate demand."
            elif rec_type == "Schedule adjustment":
                detail = f"Adjust departure time by {np.random.randint(3, 10)} minutes to optimize connections."
            elif rec_type == "Monitor crowding":
                detail = f"Station crowding may reach {np.random.randint(75, 95)}% capacity. Consider staff reallocation."
            else:  # Address potential delay
                detail = f"Weather conditions may cause {np.random.randint(5, 15)} minute delays. Prepare contingency."
            
            recommendations.append({
                "route": route,
                "time": time,
                "date": recommendation_date,
                "recommendation_type": rec_type,
                "detail": detail,
                "severity": severity,
                "confidence": f"{np.random.randint(70, 98)}%"
            })
        
        # Create DataFrame and sort by severity and time
        rec_df = pd.DataFrame(recommendations)
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        rec_df["severity_order"] = rec_df["severity"].map(severity_order)
        rec_df = rec_df.sort_values(["severity_order", "time"]).reset_index(drop=True)
        
        # Display recommendation statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Recommendations", len(rec_df))
        with col2:
            high_priority = len(rec_df[rec_df["severity"] == "High"])
            st.metric("High Priority", high_priority)
        with col3:
            routes_affected = rec_df["route"].nunique()
            st.metric("Routes Affected", routes_affected)
        
        # Display recommendations by priority
        for severity in ["High", "Medium", "Low"]:
            severity_recs = rec_df[rec_df["severity"] == severity]
            if len(severity_recs) > 0:
                st.subheader(f"{severity} Priority Recommendations")
                
                for _, row in severity_recs.iterrows():
                    # Set colors based on severity
                    severity_color = "#fff3cd" if row["severity"] == "Low" else "#ffe2e2" if row["severity"] == "High" else "#e2f0ff"
                    border_color = "#ffecb5" if row["severity"] == "Low" else "#ffcccb" if row["severity"] == "High" else "#b8daff"
                    icon = "⚠️" if row["severity"] == "Low" else "🔥" if row["severity"] == "High" else "ℹ️"
                    
                    # Display recommendation card
                    st.markdown(f"""
                    <div style='background: {severity_color}; border-left: 6px solid {border_color}; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1rem;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <div>
                                <span style='font-size:1.3rem;'>{icon}</span>
                                <b>{row["route"]}</b> at <b>{row["time"]}</b>
                            </div>
                            <div>
                                <span style='color:#666; font-size: 0.9rem;'>Confidence: {row["confidence"]}</span>
                            </div>
                        </div>
                        <p style='margin-top: 0.5rem; margin-bottom: 0; font-weight: bold;'>{row["recommendation_type"]}</p>
                        <p style='margin-top: 0.3rem; margin-bottom: 0;'>{row["detail"]}</p>
                    </div>
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