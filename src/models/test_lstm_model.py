#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_lstm_model.py: Script to test and evaluate the trained LSTM model for KomuterPulse

This script loads the best LSTM model saved during training and performs
various tests and evaluations on the test dataset.

Usage:
    python test_lstm_model.py

Author: [Your Name]
Date: May 5, 2025
"""

import os
import sys
import numpy as np  # Fixed numpy import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
sns.set_context("notebook", font_scale=1.2)

# Make sure directory for saving figures exists
os.makedirs('../models', exist_ok=True)


def load_model_and_preprocessing():
    """Load the trained model and preprocessing information."""
    
    # Use absolute path resolution to ensure files are found correctly
    CURRENT_DIR = Path(__file__).resolve().parent
    MODEL_DIR = CURRENT_DIR.parent.parent / 'models'
    print(f"Looking for models in: {MODEL_DIR}")
    
    # List all files in the model directory
    print("Available files in the model directory:")
    try:
        for file in MODEL_DIR.iterdir():
            print(f"  - {file.name}")
    except Exception as e:
        print(f"Error listing files in model directory: {e}")
    
    # Try loading the basic LSTM model first (which is indicated as the best model in the notebook)
    best_model_path = MODEL_DIR / "lstm_model_basic_lstm.h5"
    
    # If that doesn't exist, fall back to lstm_model_best.h5
    if not best_model_path.exists():
        print(f"Could not find {best_model_path}, trying alternate model file...")
        best_model_path = MODEL_DIR / "lstm_model_best.h5"
        
    preprocessing_path = MODEL_DIR / "lstm_preprocessing_info.pkl"
    
    # Check if the model file exists
    if not best_model_path.exists():
        print(f"Error: Model file not found at {best_model_path}")
        available_files = list(MODEL_DIR.glob('*.h5'))
        if available_files:
            print(f"Available model files: {[f.name for f in available_files]}")
            # Use the first available model file
            best_model_path = available_files[0]
            print(f"Using alternative model file: {best_model_path}")
        else:
            print("No .h5 files found in the models directory")
            sys.exit(1)
        
    # Check if preprocessing file exists
    if not preprocessing_path.exists():
        print(f"Error: Preprocessing info not found at {preprocessing_path}")
        available_files = list(MODEL_DIR.glob('*.pkl'))
        if available_files:
            print(f"Available pickle files: {[f.name for f in available_files]}")
        else:
            print("No .pkl files found in the models directory")
        sys.exit(1)
    
    try:
        # Load the model
        model = load_model(str(best_model_path))
        print(f"Model loaded successfully from {best_model_path}")
        
        # Load preprocessing information
        with open(preprocessing_path, 'rb') as f:
            preprocessing_info = pickle.load(f)
        
        # Extract preprocessing components
        scaler_X = preprocessing_info['scaler_X']
        scaler_y = preprocessing_info['scaler_y']
        features_to_use = preprocessing_info['features_used']
        n_steps = preprocessing_info['n_steps']
        target_col = preprocessing_info['target_col']
        
        print(f"Using sequence length of {n_steps} and {len(features_to_use)} features")
        
        return model, preprocessing_info
        
    except Exception as e:
        print(f"Error loading model or preprocessing information: {e}")
        sys.exit(1)


def load_test_data(preprocessing_info):
    """Load and preprocess the test data."""
    
    # Use absolute path resolution to ensure files are found correctly
    CURRENT_DIR = Path(__file__).resolve().parent
    DATA_DIR = CURRENT_DIR.parent.parent / 'data' / 'processed'
    test_path = DATA_DIR / 'komuter_test.csv'
    
    print(f"Looking for test data in: {test_path}")
    
    try:
        # Load test data
        test_df = pd.read_csv(test_path)
        print(f"Test data loaded with shape: {test_df.shape}")
        
        # Extract preprocessing components
        scaler_X = preprocessing_info['scaler_X']
        scaler_y = preprocessing_info['scaler_y']
        features_to_use = preprocessing_info['features_used']
        target_col = preprocessing_info['target_col']
        
        # Filter to use only the selected features from training
        X_test = test_df[features_to_use].values
        y_test = test_df[target_col].values
        
        # Scale features and target using the same scalers from training
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Create a DataFrame with additional metadata for analysis
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        return test_df, X_test_scaled, y_test_scaled
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)


def create_test_sequences(data, targets, route_ids, dates, hours, n_steps):
    """
    Create sequences for testing from the data.
    Returns sequences and metadata needed for evaluation.
    """
    X_seq, y_seq = [], []
    metadata = []  # Store route_id, date, hour for each sequence
    
    for route_id in np.unique(route_ids):
        # Get indices for this route
        route_indices = np.where(route_ids == route_id)[0]
        if len(route_indices) <= n_steps:
            continue  # Skip routes with insufficient data
            
        sorted_indices = np.sort(route_indices)  # Ensure chronological order
        
        # Create sequences for this route
        for i in range(len(sorted_indices) - n_steps):
            # Get input sequence
            seq_indices = sorted_indices[i:i+n_steps]
            sequence = data[seq_indices]
            X_seq.append(sequence)
            
            # Get target (next value after the sequence)
            target_idx = sorted_indices[i+n_steps]
            target = targets[target_idx]
            y_seq.append(target)
            
            # Store metadata
            meta = {
                'route_id': route_id,
                'date': dates[target_idx],
                'hour': hours[target_idx]
            }
            metadata.append(meta)
    
    return np.array(X_seq), np.array(y_seq), metadata


def evaluate_model(model, X_test, y_test, scaler_y):
    """Evaluate the model and calculate performance metrics."""
    print("Making predictions on test data...")
    # Get predictions with verbosity control
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform to get actual ridership values
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (np.maximum(y_test_actual, 1e-10)))) * 100  # Avoid division by zero
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
    
    print("\nModel Performance Metrics:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    return y_test_actual, y_pred_actual, metrics


def plot_test_predictions(y_true, y_pred, title, n_samples=200):
    """Plot the actual vs predicted values from test data."""
    # Sample a subset of data points to make the plot more readable
    if len(y_true) > n_samples:
        indices = np.random.choice(len(y_true), n_samples, replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    plt.figure(figsize=(14, 8))
    
    # Plot actual vs predicted
    plt.subplot(2, 1, 1)
    plt.plot(y_true_sample, color='blue', label='Actual')
    plt.plot(y_pred_sample, color='red', label='Predicted')
    plt.title(f'{title} - Actual vs Predicted Ridership (Sample)')
    plt.ylabel('Ridership')
    plt.legend()
    plt.grid(True)
    
    # Scatter plot
    plt.subplot(2, 1, 2)
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title('Prediction Scatter Plot')
    plt.xlabel('Actual Ridership')
    plt.ylabel('Predicted Ridership')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../models/test_predictions.png')
    plt.show()


def plot_error_distribution(y_true, y_pred, mae):
    """Plot histograms of prediction errors."""
    errors = y_pred - y_true
    
    plt.figure(figsize=(14, 6))
    
    # Absolute error histogram
    plt.subplot(1, 2, 1)
    plt.hist(np.abs(errors), bins=30, alpha=0.7)
    plt.axvline(mae, color='red', linestyle='dashed', linewidth=1)
    plt.text(mae*1.1, plt.ylim()[1]*0.9, f'MAE: {mae:.2f}', color='red')
    plt.title('Absolute Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    
    # Error histogram
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
    plt.title('Error Distribution')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('../models/error_distribution.png')
    plt.show()


def analyze_by_time_of_day(metadata_df):
    """Analyze model performance by time of day."""
    if 'hour' in metadata_df.columns:
        hour_performance = metadata_df.groupby('hour').agg({
            'abs_error': 'mean',
            'actual': 'mean',
            'predicted': 'mean',
            'route_id': 'count'
        }).reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.bar(hour_performance['hour'], hour_performance['abs_error'], alpha=0.7)
        plt.title('Mean Absolute Error by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Absolute Error')
        plt.xticks(range(0, 24))
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig('../models/error_by_hour.png')
        plt.show()
        
        return hour_performance
    
    return None


def analyze_by_route(metadata_df):
    """Analyze model performance by route."""
    if 'route_id' in metadata_df.columns and len(metadata_df) > 0:
        # Get top 10 routes by error
        top_routes_by_error = metadata_df.groupby('route_id').agg({
            'abs_error': 'mean',
            'actual': 'mean',
            'predicted': 'mean',
            'route_id': 'count'
        }).rename(columns={'route_id': 'count'}).sort_values('abs_error', ascending=False).head(10)
        
        print("\nTop 10 Routes with Highest Mean Absolute Error:")
        print("=" * 60)
        print(top_routes_by_error)
        
        return top_routes_by_error
    
    return None


def forecast_multiple_steps(model, initial_sequence, n_future_steps, scaler_y):
    """
    Forecast multiple steps into the future using the LSTM model.
    
    Args:
        model: Trained LSTM model
        initial_sequence: Initial sequence to start forecasting from (shape: [1, n_steps, n_features])
        n_future_steps: Number of steps to forecast
        scaler_y: Scaler used for the target variable
    
    Returns:
        forecast: Array of forecasted values in original scale
    """
    # Create a copy of the initial sequence
    curr_sequence = initial_sequence.copy()
    forecast_values = []
    
    # Get sequence dimensions
    _, n_steps, n_features = curr_sequence.shape
    
    # Forecast each step
    for _ in range(n_future_steps):
        # Predict the next value 
        next_pred = model.predict(curr_sequence, verbose=0)[0, 0]
        forecast_values.append(next_pred)
        
        # Create a new row with all features set to 0 except the first one (target)
        new_row = np.zeros((1, 1, n_features))
        new_row[0, 0, 0] = next_pred  # Set the target feature
        
        # Update sequence for next prediction by removing first timestep and adding prediction
        curr_sequence = np.concatenate([curr_sequence[:, 1:, :], new_row], axis=1)
    
    # Convert predictions back to original scale
    forecast_values = np.array(forecast_values).reshape(-1, 1)
    forecast_original = scaler_y.inverse_transform(forecast_values).flatten()
    
    return forecast_original


def plot_multi_step_forecast(forecast_values, forecast_horizon):
    """Plot multi-step forecast values."""
    plt.figure(figsize=(12, 6))
    plt.plot(range(forecast_horizon), forecast_values, marker='o', linestyle='-', color='blue', label='Forecast')
    plt.title(f'{forecast_horizon}-Hour Ridership Forecast')
    plt.xlabel('Hours into Future')
    plt.ylabel('Predicted Ridership')
    plt.grid(True)
    plt.legend()
    plt.savefig('../models/multi_step_forecast.png')
    plt.show()


def main():
    """Main function to test the LSTM model."""
    print("=" * 80)
    print("LSTM Model Testing for KomuterPulse Ridership Prediction")
    print("=" * 80)
    
    # Load model and preprocessing information
    model, preprocessing_info = load_model_and_preprocessing()
    
    # Load test data
    test_df, X_test_scaled, y_test_scaled = load_test_data(preprocessing_info)
    
    # Extract necessary information from preprocessing
    n_steps = preprocessing_info['n_steps']
    scaler_y = preprocessing_info['scaler_y']
    
    # Create test sequences
    route_ids = test_df['route_id'].values if 'route_id' in test_df.columns else np.zeros(len(test_df))
    dates = test_df['date'].values
    hours = test_df['hour'].values
    
    print("\nCreating test sequences...")
    X_test_seq, y_test_seq, test_metadata = create_test_sequences(
        X_test_scaled, y_test_scaled, route_ids, dates, hours, n_steps
    )
    print(f"Created {len(X_test_seq)} test sequences with shape: {X_test_seq.shape}")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    y_test_actual, y_test_pred, test_metrics = evaluate_model(
        model, X_test_seq, y_test_seq, scaler_y
    )
    
    # Visualize predictions
    print("\nVisualizing predictions vs actual values...")
    plot_test_predictions(y_test_actual, y_test_pred, "LSTM Model Test Results")
    
    # Plot error distribution
    print("\nAnalyzing error distribution...")
    plot_error_distribution(y_test_actual, y_test_pred, test_metrics['MAE'])
    
    # Analyze performance by time and route
    print("\nAnalyzing performance by time of day and route...")
    meta_df = pd.DataFrame(test_metadata)
    meta_df['actual'] = y_test_actual
    meta_df['predicted'] = y_test_pred
    meta_df['error'] = meta_df['predicted'] - meta_df['actual']
    meta_df['abs_error'] = np.abs(meta_df['error'])
    
    hour_performance = analyze_by_time_of_day(meta_df)
    route_performance = analyze_by_route(meta_df)
    
    # Perform multi-step forecasting
    print("\nPerforming multi-step forecasting...")
    forecast_horizon = 24  # Predict 24 hours ahead
    
    # Select a sample sequence for forecasting
    sample_idx = np.random.randint(0, len(X_test_seq))
    sample_sequence = X_test_seq[sample_idx:sample_idx+1]
    
    # Forecast multiple steps into the future
    forecast_values = forecast_multiple_steps(model, sample_sequence, forecast_horizon, scaler_y)
    
    print(f"Forecasted values for the next {forecast_horizon} hours:")
    for i, value in enumerate(forecast_values):
        print(f"Hour {i+1}: {value:.2f}")
    
    # Plot multi-step forecast
    plot_multi_step_forecast(forecast_values, forecast_horizon)
    
    # Save test results
    print("\nSaving test results...")
    test_results = {
        'metrics': test_metrics,
        'test_size': len(y_test_seq),
        'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save test results to a file - use absolute path instead of relative path
    CURRENT_DIR = Path(__file__).resolve().parent
    MODEL_DIR = CURRENT_DIR.parent.parent / 'models'
    results_path = MODEL_DIR / "lstm_test_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(test_results, f)
    
    print(f"Test results saved to {results_path}")
    print("\nModel testing completed successfully!")


if __name__ == "__main__":
    main()