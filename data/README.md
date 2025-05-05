# KTM Komuter Data Files

This directory contains data files for the KTM Komuter ML project.

## Data Structure

- `raw/`: Contains original, unmodified data files
- `processed/`: Contains preprocessed data files for model development

## Large Data Files

The processed data files exceed GitHub's file size limits (over 50MB and 100MB) and are excluded from version control with `.gitignore`. These files include:

- `komuter_processed.csv` (301.89 MB) - Complete dataset with all features
- `komuter_train.csv` (241.78 MB) - Training dataset (80% of data)
- `komuter_test.csv` (60.11 MB) - Testing dataset (20% of data)
- `komuter_features.csv` (69.81 MB) - Transaction-level dataset with features

## Processed Dataset Overview

### komuter_features.csv
- Transaction-level dataset with all original records and engineered features
- Preserves individual passenger journey details
- Contains time segment features (peak hours, weekends) and outlier flags
- Suitable for detailed journey analysis and anomaly detection

### komuter_processed.csv
- Aggregated route-hour level dataset combining records by route, date, and hour
- Includes summary statistics (total/avg/max ridership) for each route-hour
- Features time series elements like lagged variables and rolling statistics
- Optimized for route scheduling and ridership forecasting

### komuter_train.csv
- Chronological subset (first ~80%) of processed data for model training
- Contains all features from the processed dataset
- Used for developing forecasting and anomaly detection models

### komuter_test.csv
- Chronological subset (last ~20%) of processed data for model evaluation
- Ensures proper temporal validation without data leakage
- Simulates real-world forecasting scenarios with future data

## Getting the Data Files

Team members can obtain these files through any of these options:

### Option 1: Regenerate from raw data

Run the data preprocessing notebooks in sequence:
1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_preprocessing.ipynb`

### Option 2: Download from shared storage

Team members can download the processed files from our shared Google Drive folder:
[KTM Komuter Data Files](https://drive.google.com/drive/folders/1uTp05tN6uCVXIlCNIDzUw0r4d0HBMSsy?usp=drive_link)