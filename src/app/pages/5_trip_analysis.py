import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from src.app.utils.model_loader import ModelManager
from src.app.utils.data_processor import process_user_input

st.set_page_config(page_title="Trip Analysis", page_icon="📍")

st.title("📍 Trip Analysis & Prediction")
st.markdown("Select your trip details to get ridership predictions and insights.")

# --- Model Loading ---
# Do NOT cache a method with 'self' as an argument. Instead, cache a function that returns the loaded model manager.
@st.cache_resource
def get_model_manager():
    manager = ModelManager()
    manager.load_models()  # Do not cache this method, just call it
    return manager

model_manager = get_model_manager()

if model_manager is None or model_manager.lstm_model is None or model_manager.preprocessing_info is None:
    st.error("Failed to load model or preprocessing information. Please check the 'models' directory.")
else:
    # --- Trip Input Form ---
    st.subheader("Enter Trip Details")

    # Mock list of stations (replace with actual data loading later)
    # Ideally, load unique stations from your training data using the preprocessing_info if available
    # For now, keep the mock list:
    stations = ["KL Sentral", "Bank Negara", "Segambut", "Kepong", "Tanjung Malim", "Gemas"]

    col1, col2 = st.columns(2)

    with col1:
        origin_station = st.selectbox("Origin Station", stations)

    with col2:
        destination_station = st.selectbox("Destination Station", stations)

    date_input = st.date_input("Date", datetime.now())
    time_input = st.time_input("Time", datetime.now())

    # --- Prediction Button ---
    if st.button("Get Prediction"):
        # Perform basic validation
        if origin_station == destination_station:
            st.warning("Origin and Destination stations cannot be the same.")
        else:
            st.info("Processing input and generating prediction...")
            
            try:
                # Process user input
                processed_input = process_user_input(origin_station, destination_station, date_input, time_input, model_manager)
                
                if processed_input is not None:
                    # Make prediction
                    prediction = model_manager.predict(processed_input)
                    
                    # --- Display Prediction Result ---
                    if prediction is not None:
                        st.subheader("Prediction Result")
                        # Assuming prediction is a single number (ridership count)
                        st.success(f"Predicted Ridership: **{prediction:.0f}**")
                        st.markdown("*(Note: This is a raw prediction based on available features from input. Anomaly detection and detailed analysis would require additional logic and historical data lookup.)*")
                    else:
                        st.error("Failed to get prediction from the model.")
                else:
                    st.error("Failed to process user input.")
                    
            except Exception as e:
                st.error(f"An error occurred during processing or prediction: {e}")

    # --- Placeholder for additional analysis/visualizations ---
    st.markdown("---")
    st.subheader("Additional Analysis (Coming Soon)")
    st.info("This section will include anomaly detection results, historical trends, and other relevant visualizations for the selected trip.") 