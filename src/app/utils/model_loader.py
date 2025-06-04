# src/app/utils/model_loader.py

import tensorflow as tf
import pickle
import numpy as np
import streamlit as st
import os # Import os for path joining

class ModelManager:
    def __init__(self):
        self.lstm_model = None
        self.preprocessing_info = None
        # Assuming the models and pkl are in the project's 'models' directory
        self.model_path = os.path.join('models', 'lstm_model_best.h5')
        self.preprocessing_info_path = os.path.join('models', 'lstm_preprocessing_info.pkl')

    def load_models(self):
        """Load the trained models and preprocessing information"""
        st.info("Loading model and preprocessing info...")
        try:
            # Load the trained LSTM model
            self.lstm_model = tf.keras.models.load_model(self.model_path)
            st.success("LSTM model loaded successfully.")

            # Load the preprocessing information
            with open(self.preprocessing_info_path, 'rb') as f:
                self.preprocessing_info = pickle.load(f)
            st.success("Preprocessing information loaded successfully.")

            # Check for expected preprocessing objects (e.g., scalers, maps)
            if 'scaler_X' not in self.preprocessing_info:
                 st.warning("'scaler_X' not found in preprocessing_info. Scaling might be skipped in data processing.")
            if 'scaler_y' not in self.preprocessing_info:
                 st.warning("'scaler_y' (for inverse transform) not found in preprocessing_info. Inverse scaling might be skipped in prediction.")
            if 'origin_freq_map' not in self.preprocessing_info or 'dest_freq_map' not in self.preprocessing_info:
                 st.warning("Frequency maps not found in preprocessing_info. Frequency encoding might not work correctly.")
            if 'origin_pop_map' not in self.preprocessing_info or 'dest_pop_map' not in self.preprocessing_info:
                 st.warning("Popularity maps not found in preprocessing_info. Popularity encoding might not work correctly.")

            return True
        except FileNotFoundError as e:
            st.error(f"Error loading file: {e}. Make sure 'models/lstm_model_best.h5' and 'models/lstm_preprocessing_info.pkl' exist.")
            return False
        except Exception as e:
            st.error(f"Error loading models or preprocessing info: {str(e)}")
            return False

    def predict(self, processed_input):
        """Make predictions using the loaded model and inverse transform if scaler_y is available."""
        if self.lstm_model is None:
            st.error("Model not loaded. Cannot make prediction.")
            return None

        st.info("Making prediction...")
        try:
            # Make prediction with the LSTM model
            prediction_scaled = self.lstm_model.predict(processed_input)

            # Inverse transform the prediction if the target scaler is available
            scaler_y = self.preprocessing_info.get('scaler_y')
            if scaler_y is not None:
                # Prediction is likely in shape (1, 1) or (1,). Need to reshape for inverse_transform
                prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
                st.success("Prediction inverse-scaled.")
            else:
                prediction = prediction_scaled.flatten()[0]
                st.warning("scaler_y not found in preprocessing_info. Prediction is not inverse-scaled.")

            st.success("Prediction generated.")
            # Ensure prediction is a standard number, not a numpy array
            return float(prediction)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None

# Instantiate the ModelManager (can be done once and reused)
# model_manager = ModelManager() # Avoid instantiating directly here if using st.cache_resource on methods 