# src/app/utils/model_loader.py

import tensorflow as tf
import pickle
import numpy as np
import streamlit as st
import os # Import os for path joining
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.lstm_model = None
        self.preprocessing_info = None
        # Assuming the models and pkl are in the project's 'models' directory
        self.model_path = os.path.join('models', 'lstm_model_best.h5')
        self.preprocessing_info_path = os.path.join('models', 'lstm_preprocessing_info.pkl')
        logger.info("ModelManager initialized with paths: %s and %s", 
                   self.model_path, self.preprocessing_info_path)

    def load_models(self):
        """Load the trained models and preprocessing information"""
        st.info("Loading model and preprocessing info...")
        logger.info("Starting to load models and preprocessing info")
        
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                error_msg = f"Model file not found at path: {self.model_path}"
                logger.error(error_msg)
                st.error(error_msg)
                return False
                
            # Check if preprocessing info file exists
            if not os.path.exists(self.preprocessing_info_path):
                error_msg = f"Preprocessing info file not found at path: {self.preprocessing_info_path}"
                logger.error(error_msg)
                st.error(error_msg)
                return False
            
            # Load the trained LSTM model
            logger.info("Loading LSTM model from %s", self.model_path)
            self.lstm_model = tf.keras.models.load_model(self.model_path)
            st.success("LSTM model loaded successfully.")
            logger.info("LSTM model loaded successfully")

            # Load the preprocessing information
            logger.info("Loading preprocessing info from %s", self.preprocessing_info_path)
            with open(self.preprocessing_info_path, 'rb') as f:
                self.preprocessing_info = pickle.load(f)
            st.success("Preprocessing information loaded successfully.")
            logger.info("Preprocessing information loaded successfully")

            # Check for expected preprocessing objects (e.g., scalers, maps)
            if 'scaler_X' not in self.preprocessing_info:
                logger.warning("'scaler_X' not found in preprocessing_info")
                st.warning("'scaler_X' not found in preprocessing_info. Scaling might be skipped in data processing.")
            if 'scaler_y' not in self.preprocessing_info:
                logger.warning("'scaler_y' not found in preprocessing_info")
                st.warning("'scaler_y' (for inverse transform) not found in preprocessing_info. Inverse scaling might be skipped in prediction.")
            if 'origin_freq_map' not in self.preprocessing_info or 'dest_freq_map' not in self.preprocessing_info:
                logger.warning("Frequency maps not found in preprocessing_info")
                st.warning("Frequency maps not found in preprocessing_info. Frequency encoding might not work correctly.")
            if 'origin_pop_map' not in self.preprocessing_info or 'dest_pop_map' not in self.preprocessing_info:
                logger.warning("Popularity maps not found in preprocessing_info")
                st.warning("Popularity maps not found in preprocessing_info. Popularity encoding might not work correctly.")

            # Log model summary
            logger.info("Model summary: %s", self.lstm_model.summary())
            
            return True
        except FileNotFoundError as e:
            error_msg = f"Error loading file: {e}. Make sure 'models/lstm_model_best.h5' and 'models/lstm_preprocessing_info.pkl' exist."
            logger.error(error_msg)
            st.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Error loading models or preprocessing info: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            return False

    def predict(self, processed_input):
        """Make predictions using the loaded model and inverse transform if scaler_y is available."""
        if self.lstm_model is None:
            error_msg = "Model not loaded. Cannot make prediction."
            logger.error(error_msg)
            st.error(error_msg)
            return None

        st.info("Making prediction...")
        logger.info("Making prediction with input shape: %s", processed_input.shape if hasattr(processed_input, 'shape') else "unknown")
        
        try:
            # Make prediction with the LSTM model
            prediction_scaled = self.lstm_model.predict(processed_input)
            logger.info("Raw prediction result: %s", prediction_scaled)

            # Inverse transform the prediction if the target scaler is available
            scaler_y = self.preprocessing_info.get('scaler_y')
            if scaler_y is not None:
                # Prediction is likely in shape (1, 1) or (1,). Need to reshape for inverse_transform
                prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
                logger.info("Prediction after inverse scaling: %s", prediction)
                st.success("Prediction inverse-scaled.")
            else:
                prediction = prediction_scaled.flatten()[0]
                logger.warning("scaler_y not found in preprocessing_info. Prediction is not inverse-scaled: %s", prediction)
                st.warning("scaler_y not found in preprocessing_info. Prediction is not inverse-scaled.")

            st.success("Prediction generated.")
            # Ensure prediction is a standard number, not a numpy array
            return float(prediction)
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            return None

# Instantiate the ModelManager (can be done once and reused)
# model_manager = ModelManager() # Avoid instantiating directly here if using st.cache_resource on methods 