# src/app/utils/data_processor.py

# src/app/utils/data_processor.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .model_loader import ModelManager

logger = logging.getLogger(__name__)

def get_historical_ridership_data(route, end_datetime, hours_back=24):
    """
    Get historical ridership data for a route.
    This is a simplified version - in production, this would query a database.
    For now, we'll use the recommendations data as a proxy for historical patterns.
    """
    try:
        # In a real implementation, this would query historical ridership data
        # For now, we'll simulate based on typical patterns
        
        # Generate synthetic historical data based on typical daily patterns
        dates = pd.date_range(end=end_datetime, periods=hours_back, freq='h')
        
        # Create realistic hourly patterns (higher during rush hours)
        ridership_pattern = []
        for dt in dates:
            hour = dt.hour
            dow = dt.dayofweek
            
            # Base ridership
            base = 50
            
            # Rush hour multipliers
            if hour in [7, 8, 17, 18, 19]:  # Rush hours
                multiplier = 2.5
            elif hour in [6, 9, 16, 20]:  # Near rush hours
                multiplier = 1.8
            elif hour in [10, 11, 12, 13, 14, 15]:  # Business hours
                multiplier = 1.3
            elif hour in [21, 22]:  # Evening
                multiplier = 0.8
            else:  # Night/early morning
                multiplier = 0.3
            
            # Weekend adjustment
            if dow >= 5:  # Weekend
                if hour in [10, 11, 12, 13, 14, 15, 16]:  # Weekend active hours
                    multiplier *= 1.2
                else:
                    multiplier *= 0.7
            
            # Add some randomness
            noise = np.random.normal(1, 0.2)
            ridership = max(1, base * multiplier * noise)
            ridership_pattern.append(ridership)
        
        # Create DataFrame
        historical_data = pd.DataFrame({
            'datetime': dates,
            'ridership': ridership_pattern,
            'route': route
        })
        
        return historical_data
        
    except Exception as e:
        logger.error(f"Error generating historical data: {e}")
        return None

def create_lstm_features(historical_data):
    """
    Create the exact features that the LSTM model expects.
    Based on the trained model's feature list.
    """
    try:
        df = historical_data.copy()
        df = df.sort_values('datetime')
        
        # Calculate the features that the model expects
        df['avg_ridership'] = df['ridership'].rolling(window=6, min_periods=1).mean()
        df['max_ridership'] = df['ridership'].rolling(window=12, min_periods=1).max()
        
        # Differences
        df['ridership_diff_1h'] = df['ridership'].diff(1)
        df['ridership_diff_2h'] = df['ridership'].diff(2)
        df['ridership_diff_1d'] = df['ridership'].diff(24) if len(df) >= 24 else 0
        df['ridership_diff_1w'] = df['ridership'].diff(168) if len(df) >= 168 else 0  # 7*24 hours
        
        # Percentage changes
        df['ridership_pct_change_1d'] = df['ridership'].pct_change(24) if len(df) >= 24 else 0
        df['ridership_pct_change_1w'] = df['ridership'].pct_change(168) if len(df) >= 168 else 0
        
        # Rolling statistics
        df['rolling_mean_3h'] = df['ridership'].rolling(window=3, min_periods=1).mean()
        df['rolling_mean_6h'] = df['ridership'].rolling(window=6, min_periods=1).mean()
        df['rolling_max_3h'] = df['ridership'].rolling(window=3, min_periods=1).max()
        df['rolling_max_6h'] = df['ridership'].rolling(window=6, min_periods=1).max()
        df['rolling_max_12h'] = df['ridership'].rolling(window=12, min_periods=1).max()
        df['rolling_max_24h'] = df['ridership'].rolling(window=24, min_periods=1).max()
        df['rolling_min_3h'] = df['ridership'].rolling(window=3, min_periods=1).min()
        df['rolling_std_3h'] = df['ridership'].rolling(window=3, min_periods=1).std()
        df['rolling_std_6h'] = df['ridership'].rolling(window=6, min_periods=1).std()
        
        # Lag features
        df['total_ridership_lag_2h'] = df['ridership'].shift(2)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating LSTM features: {e}")
        return None

def process_user_input_for_lstm(origin, destination, date, time, model_manager: ModelManager):
    """
    Process user input for LSTM prediction using historical data approach.
    """
    try:
        # Combine date and time
        target_datetime = datetime.combine(date, time)
        route = f"{origin} → {destination}"
        
        # Get historical data (24 hours before the target time)
        historical_data = get_historical_ridership_data(route, target_datetime, hours_back=24)
        
        if historical_data is None:
            logger.error("Failed to get historical data")
            return None
        
        # Create LSTM features
        feature_df = create_lstm_features(historical_data)
        
        if feature_df is None:
            logger.error("Failed to create LSTM features")
            return None
        
        # Select the exact features the model expects
        expected_features = [
            'avg_ridership', 'max_ridership', 'ridership_diff_1d', 'ridership_diff_1h', 
            'ridership_diff_1w', 'ridership_diff_2h', 'ridership_pct_change_1d', 
            'ridership_pct_change_1w', 'rolling_max_12h', 'rolling_max_24h', 
            'rolling_max_3h', 'rolling_max_6h', 'rolling_mean_3h', 'rolling_mean_6h', 
            'rolling_min_3h', 'rolling_std_3h', 'rolling_std_6h', 'total_ridership_lag_2h'
        ]
        
        # Ensure all features exist
        for feature in expected_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0
        
        # Select features in the correct order
        feature_matrix = feature_df[expected_features].values
        
        # Get the last 24 time steps for prediction
        if len(feature_matrix) >= 24:
            lstm_input = feature_matrix[-24:]  # Last 24 hours
        else:
            # Pad with zeros if we don't have enough data
            padding = np.zeros((24 - len(feature_matrix), len(expected_features)))
            lstm_input = np.vstack([padding, feature_matrix])
        
        # Apply scaling
        preprocessing_info = model_manager.preprocessing_info
        scaler_X = preprocessing_info.get('scaler_X')
        
        if scaler_X is not None:
            # Reshape for scaling (samples, features)
            original_shape = lstm_input.shape
            lstm_input_scaled = scaler_X.transform(lstm_input.reshape(-1, lstm_input.shape[-1]))
            lstm_input_scaled = lstm_input_scaled.reshape(original_shape)
        else:
            lstm_input_scaled = lstm_input
            logger.warning("No scaler found, using unscaled data")
        
        # Reshape for LSTM: (batch_size, time_steps, features)
        lstm_input_final = lstm_input_scaled.reshape(1, 24, len(expected_features))
        
        logger.info(f"LSTM input shape: {lstm_input_final.shape}")
        return lstm_input_final
        
    except Exception as e:
        logger.error(f"Error processing user input for LSTM: {e}")
        return None

def mock_process_user_input(origin, destination, date, time):
    """A mock data processor for initial testing."""
    logger.info(f"Mock processing input: {origin} to {destination} on {date} at {time}")
    # Return mock data in the correct shape for LSTM (1, 24, 18)
    return np.random.rand(1, 24, 18)