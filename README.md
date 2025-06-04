# KomuterPulse: Real-time Transit Intelligence Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

KomuterPulse is an advanced machine learning project developed for the WIA1006 Machine Learning course at FCSIT, University of Malaya. This transit intelligence platform transforms raw ridership data from KTM Komuter services into actionable insights through time series forecasting and anomaly detection.

## Project Objective

We are developing a comprehensive real-time transit intelligence platform that transforms raw ridership data into actionable insights for KTM Komuter operations. KomuterPulse combines advanced time series forecasting with anomaly detection using a hybrid AI approach to revolutionize transit management.

### Dataset

The project leverages public transportation data from the Malaysian government open data initiative:

* **Dataset**: [Hourly Origin-Destination Ridership for KTM Komuter](https://data.gov.my/data-catalogue/ridership_od_komuter?)
* **Volume**: 670000+ records
* **Format**: Time series data with origin-destination pairs

### Target Variables
- **Primary**: Hourly ridership between station pairs (regression)
- **Secondary**: Anomaly classification (binary: normal vs. unusual patterns)

### Key Features

#### Time-Based Route Importance
- Predict the relative importance of each route by hour with visual heatmaps
- Create a dynamic "heat map" of the network showing where resources should be allocated
- Identify critical routes that require prioritization during specific time periods
- Provide real-time passenger load predictions to prevent overcrowding

#### Anomaly Detection & Predictive Intelligence
- Identify unusual ridership patterns that deviate from expected norms
- Flag situations where additional trains may be needed or schedule adjustments required
- Detect potential service disruptions before they impact passenger experience
- Generate predictive alerts for station managers and operations teams

#### Actionable Schedule Recommendations
- Convert predictions into concrete operational recommendations
- Optimize departure frequency by hour and station
- Determine when to deploy additional train cars
- Identify days that might require extended operating hours
- Suggest dynamic pricing strategies based on demand forecasting

#### Environmental & Social Impact Assessment
- Calculate carbon footprint reduction metrics from optimized scheduling
- Provide accessibility scoring to highlight stations needing improvement
- Analyze multi-modal integration opportunities with other transit systems
- Quantify social impact through improved service reliability metrics

#### Advanced Visualization Suite
- Interactive network diagrams showing passenger flows
- Real-time operational dashboards with predictive alerts
- Animated time-series visualizations showing historical patterns
- Comparative analysis of actual vs. optimized schedules

### Technical Innovation
Our solution leverages cutting-edge techniques:
- Hybrid modeling approach combining statistical methods with deep learning
- Transfer learning techniques adapted from other transit systems
- Explainable AI components making predictions transparent and trustworthy
- Edge deployment capabilities for station-level processing and real-time insights

### Business Value
KomuterPulse will provide KTM with measurable benefits:
- Projected revenue increases of 15-20% from optimized scheduling and dynamic pricing
- Operational cost savings through more efficient resource allocation
- Customer satisfaction improvements from reduced delays and overcrowding
- Sustainability metrics showing reduced emissions from optimized train deployment
- Data-driven decision making for both daily operations and strategic planning
- Enhanced ability to plan for special events and holidays

## Solution Architecture

KomuterPulse combines Long Short-Term Memory (LSTM) neural networks with classical machine learning techniques to deliver a comprehensive transit intelligence solution. Our system:

1. **Processes** historical ridership data
2. **Analyzes** temporal patterns and anomalies
3. **Forecasts** future ridership demand
4. **Recommends** operational optimizations

### Core Capabilities

1. **Time-Based Route Importance**
   - Predictive heatmaps of network demand
   - Resource allocation optimization
   - Peak demand forecasting

2. **Anomaly Detection & Predictive Intelligence**
   - Real-time pattern deviation detection
   - Proactive service disruption alerts
   - Operational anomaly classification

3. **Actionable Schedule Recommendations**
   - Data-driven departure frequency optimization
   - Dynamic capacity planning
   - Demand-based resource allocation

4. **Advanced Visualization Suite**
   - Interactive network flow diagrams
   - Temporal pattern dashboards
   - Comparative performance analytics

## Technical Implementation

### Machine Learning Pipeline

Our solution implements an end-to-end machine learning pipeline:

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

### Model Architecture

The core of our system uses a multi-layered LSTM architecture optimized for time series forecasting:

- **Input Layer**: Sequential time windows of ridership patterns
- **Hidden Layers**: Multiple LSTM layers with dropout for regularization
- **Output Layer**: Regression predictions for future ridership

### Evaluation Metrics

Performance is measured using industry-standard metrics:

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Square Error for prediction accuracy |
| MAE | Mean Absolute Error for absolute differences |
| R² | Coefficient of determination for explained variance |
| MAPE | Mean Absolute Percentage Error for relative performance |

## Current Project Structure

```
KomuterPulse/
├── README.md                       # Project documentation
├── requirements.txt                # Package dependencies
├── view_pickle_files.py            # Utility to inspect serialized data
├── WIA1006 Group Assignment 2024_25.pdf  # Assignment specifications
├── data/
│   ├── README.md                   # Data documentation
│   ├── processed/                  # Processed datasets
│   │   ├── feature_subsets.pkl     # Serialized feature groups
│   │   ├── komuter_features.csv    # Feature-engineered data (69.81 MB)
│   │   ├── komuter_processed.csv   # Fully processed dataset (301.89 MB)
│   │   ├── komuter_test.csv        # Testing dataset (60.11 MB)
│   │   └── komuter_train.csv       # Training dataset (241.78 MB)
│   └── raw/
│       └── komuter_2025.csv        # Original dataset
├── models/
│   ├── lstm_model_basic_lstm.h5    # Trained basic LSTM model
│   ├── lstm_model_best.h5          # Best performing model
│   ├── lstm_model_summary.pkl      # Model performance metrics
│   └── lstm_preprocessing_info.pkl # Preprocessing parameters
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Initial data analysis
│   ├── 02_data_preprocessing.ipynb # Data cleaning and preparation
│   ├── 03_feature_engineering.ipynb # Feature creation and selection
│   ├── 04_model_development.ipynb  # Model building and training
│   └── 05_model_evaluation.ipynb   # Performance assessment
└── src/
    ├── Introduction.py             # Project introduction script
    └── data/
        ├── data_loading.py         # Data import utilities
        └── make_dataset.py         # Dataset creation scripts
```

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Matplotlib, Scikit-learn
- See `requirements.txt` for complete dependencies

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MarcusMQF/komuter-ml-analysis.git
   cd 
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Jupyter Notebooks
The analysis is organized as sequential Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

Begin with `01_data_exploration.ipynb` and follow the numbered sequence.

#### Alternative: Google Colab
You can also run the notebooks in Google Colab by uploading them from this repository.

## Acknowledgments

- Faculty of Computer Science & Information Technology, University of Malaya
- Malaysian government's open data initiative
- KTM Komuter for the dataset

## How to run
```
python -m streamlit run src/app/main.py
```

## Team: Artificial Not Intelligent

- Mah Qing Fung
- Ajax Kang AJ
- Chong Yu En
- Lee Yi Mei
- Oi Kay Yi