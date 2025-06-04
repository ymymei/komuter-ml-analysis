# src/app/utils/data_processor.py

import pandas as pd
import numpy as np
from datetime import datetime
from .model_loader import ModelManager # Import ModelManager from the same package

def process_user_input(origin, destination, date, time, model_manager: ModelManager):
    """Processes user input into a feature vector for model prediction."""

    # Combine date and time into a single datetime object
    datetime_obj = datetime.combine(date, time)

    # Create a dictionary with raw features
    raw_data = {
        'origin': [origin],
        'destination': [destination],
        'datetime': [datetime_obj]
    }
    input_df = pd.DataFrame(raw_data)

    # --- Feature Engineering (based on notebooks) ---
    input_df['route'] = input_df['origin'] + ' → ' + input_df['destination']
    input_df['year'] = input_df['datetime'].dt.year
    input_df['month'] = input_df['datetime'].dt.month
    input_df['day'] = input_df['datetime'].dt.day
    input_df['day_of_week'] = input_df['datetime'].dt.dayofweek
    input_df['hour'] = input_df['datetime'].dt.hour

    # Cyclical features
    input_df['hour_sin'] = np.sin(2 * np.pi * input_df['hour']/24)
    input_df['hour_cos'] = np.cos(2 * np.pi * input_df['hour']/24)
    input_df['day_of_week_sin'] = np.sin(2 * np.pi * input_df['day_of_week']/7)
    input_df['day_of_week_cos'] = np.cos(2 * np.pi * input_df['day_of_week']/7)

    # Time segment features (simplified - assuming non-weekend for rush/peak hours)
    input_df['is_weekend'] = (input_df['day_of_week'] >= 5).astype(int)
    # Note: is_rush_hour, is_peak_morning, is_peak_evening, is_business_hours, is_night_hours
    # These require considering is_weekend and hour ranges. Implementing a simplified version:
    hour = input_df['hour'].iloc[0]
    is_weekend = input_df['is_weekend'].iloc[0]
    input_df['is_rush_hour'] = int(((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19)) and not is_weekend)
    input_df['is_peak_morning'] = int(((hour >= 6 and hour <= 9) and not is_weekend))
    input_df['is_peak_evening'] = int(((hour >= 17 and hour <= 20) and not is_weekend))
    input_df['is_business_hours'] = int(((hour >= 9 and hour <= 17) and not is_weekend))
    input_df['is_night_hours'] = int(((hour >= 21 or hour <= 5)))
    input_df['hour_of_day'] = hour # Based on notebook

    # --- Encoding and Scaling (using loaded preprocessing info) ---
    # Need to access loaded maps and scalers from ModelManager
    preprocessing_info = model_manager.preprocessing_info

    if preprocessing_info is None:
        raise ValueError("Preprocessing information not loaded. Cannot process input.")

    # Apply frequency and popularity encoding
    input_df['origin_freq'] = input_df['origin'].map(preprocessing_info.get('origin_freq_map', {None:0})).fillna(0)
    input_df['destination_freq'] = input_df['destination'].map(preprocessing_info.get('dest_freq_map', {None:0})).fillna(0)
    input_df['origin_popularity'] = input_df['origin'].map(preprocessing_info.get('origin_pop_map', {None:0})).fillna(0)
    input_df['destination_popularity'] = input_df['destination'].map(preprocessing_info.get('dest_pop_map', {None:0})).fillna(0)

    # Handle potential new stations not in training data maps by assigning 0 or a default.
    # The .fillna(0) handles cases where the map doesn't have the key.

    # --- Select features for the model --- (Based on typical features used in notebooks)
    # Need to ensure the order matches the training data order
    # This requires knowing the exact list and order of features used for training the LSTM model
    # For now, let's define a likely set based on notebook analysis:
    feature_columns = [
        # Time Features
        'hour', 'day_of_week', 'year', 'month', 'day', # Include raw time features
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'is_weekend', 'is_rush_hour', 'is_peak_morning', 'is_peak_evening', 
        'is_business_hours', 'is_night_hours', 'hour_of_day',
        # Station Features
        'origin_freq', 'destination_freq', 'origin_popularity', 'destination_popularity',
        # Note: Lagged features, rolling stats, diffs, outliers are omitted for single point prediction
        # as they depend on historical context not available from user input alone.
        # If your model requires these, you'll need a data lookup mechanism.
    ]
    
    # Ensure all expected feature columns are present, add missing ones with default values (e.g., 0 or mean)
    # This part requires knowing the *exact* list of features your trained model expects.
    # For demonstration, I'll select a subset we know how to create:
    processed_features_df = input_df[[
        'hour', 'day_of_week', 'year', 'month', 'day',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'is_weekend', 'is_rush_hour', 'is_peak_morning', 'is_peak_evening', 
        'is_business_hours', 'is_night_hours', 'hour_of_day',
        'origin_freq', 'destination_freq', 'origin_popularity', 'destination_popularity'
        # ... add other features if you can generate them or fill with defaults ...
    ]]

    # Apply scaling using the scaler loaded by ModelManager
    scaler_X = preprocessing_info.get('scaler_X')
    if scaler_X is None:
         # Handle case where scaler_X was not found in the pkl. Maybe it was saved under a different key?
         # Or maybe the model expects unscaled input for some features?
         print("Warning: scaler_X not found in preprocessing_info. Skipping scaling.")
         processed_scaled_input = processed_features_df.values # Use unscaled values for now
    else:
        processed_scaled_input = scaler_X.transform(processed_features_df)
        
    # LSTM models typically expect input in the shape [samples, time_steps, features]
    # For a single prediction, samples=1, time_steps=1 (if predicting one step ahead based on current features)
    # Assuming your model predicts based on a single time step of features:
    # The shape would be [1, 1, number_of_features]
    
    # Number of features should match the model's expected input shape.
    # This is a critical point and depends on how your LSTM was trained.
    # If your LSTM expects a sequence (e.g., the last N hours), this processing logic needs to be more complex
    # to look up or simulate that sequence data.
    
    # Assuming for now it takes a single time step of features:
    # We need to know the exact number and order of features the model expects from preprocessing_info or model summary.
    # Let's check the shape after scaling - it should be (1, num_features)
    num_features = processed_scaled_input.shape[1]
    
    # Reshape for LSTM [samples, time_steps, features]
    # Assuming time_steps = 1 for a single point prediction
    lstm_input = processed_scaled_input.reshape(1, 1, num_features)

    return lstm_input

def mock_process_user_input(origin, destination, date, time):
    """A mock data processor for initial testing."""
    print(f"Mock processing input: {origin} to {destination} on {date} at {time}")
    # Simulate some processing
    import time
    time.sleep(0.2)
    return np.random.rand(1, 1, 20) # Return a mock feature array shape (1, 1, num_features) 