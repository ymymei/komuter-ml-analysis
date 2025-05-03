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

## Getting the Data Files

Team members can obtain these files through any of these options:

### Option 1: Regenerate from raw data

Run the data preprocessing notebooks in sequence:
1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_preprocessing.ipynb`

### Option 2: Download from shared storage

Team members can download the processed files from our shared Google Drive folder:
[KTM Komuter Data Files](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)

### Option 3: Request from team members

Contact any team member to share the files directly.