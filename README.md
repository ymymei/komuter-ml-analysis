# WIA1006 Machine Learning Assignment

## Project Overview
This repository contains our group's machine learning project for the WIA1006 Machine Learning course at FCSIT UM. Our goal is to apply machine learning techniques to solve real-world problems using public datasets from the [Malaysian government](https://data.gov.my/). 

## Team: Artificial Not Intelligent
- Mah Qing Fung
- Ajax Kang AJ
- Chong Yu En
- Lee Yi Mei
- Oi Kay Yi

## Theme Selection
We have selected the following dataset for our project:

### Hourly Origin-Destination Ridership: Komuter
- **Dataset Source**: [data.gov.my - Ridership OD Komuter](https://data.gov.my/data-catalogue/ridership_od_komuter?)
- **Data Volume**: ~30,000+ records

## Project Objective: Dynamic Route Scheduling & Anomaly Detection System / KomuterPulse - Real-time Transit Intelligence Platform

We are developing a comprehensive real-time transit intelligence platform that transforms raw ridership data into actionable insights for KTM Komuter operations. KomuterPulse combines advanced time series forecasting with anomaly detection using a hybrid AI approach to revolutionize transit management.

### Target Variables
- **Primary**: Hourly ridership between station pairs (regression)
- **Secondary**: Anomaly classification (binary: normal vs. unusual patterns)

### Key Features
1. **Time-Based Route Importance**:
   - Predict the relative importance of each route by hour with visual heatmaps
   - Create a dynamic "heat map" of the network showing where resources should be allocated
   - Identify critical routes that require prioritization during specific time periods
   - Provide real-time passenger load predictions to prevent overcrowding

2. **Anomaly Detection & Predictive Intelligence**:
   - Identify unusual ridership patterns that deviate from expected norms
   - Flag situations where additional trains may be needed or schedule adjustments required
   - Detect potential service disruptions before they impact passenger experience
   - Generate predictive alerts for station managers and operations teams

3. **Actionable Schedule Recommendations**:
   - Convert predictions into concrete operational recommendations
   - Optimize departure frequency by hour and station
   - Determine when to deploy additional train cars
   - Identify days that might require extended operating hours
   - Suggest dynamic pricing strategies based on demand forecasting

4. **Environmental & Social Impact Assessment**:
   - Calculate carbon footprint reduction metrics from optimized scheduling
   - Provide accessibility scoring to highlight stations needing improvement
   - Analyze multi-modal integration opportunities with other transit systems
   - Quantify social impact through improved service reliability metrics

5. **Advanced Visualization Suite**:
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

## Data Processing Pipeline

We've developed a comprehensive data processing pipeline that transforms raw Komuter ridership data into feature-rich datasets optimized for machine learning:

### Processed Dataset Overview

1. **komuter_features.csv**:
   - Transaction-level dataset with all original records and engineered features
   - Preserves individual passenger journey details
   - Contains time segment features (peak hours, weekends) and outlier flags
   - Suitable for detailed journey analysis and anomaly detection

2. **komuter_processed.csv**:
   - Aggregated route-hour level dataset combining records by route, date, and hour
   - Includes summary statistics (total/avg/max ridership) for each route-hour
   - Features time series elements like lagged variables and rolling statistics
   - Optimized for route scheduling and ridership forecasting

3. **komuter_train.csv**:
   - Chronological subset (first ~80%) of processed data for model training
   - Contains all features from the processed dataset
   - Used for developing forecasting and anomaly detection models

4. **komuter_test.csv**:
   - Chronological subset (last ~20%) of processed data for model evaluation
   - Ensures proper temporal validation without data leakage
   - Simulates real-world forecasting scenarios with future data

## Team Member Roles

### Member 1: Project Manager & Data Collection Lead
- **Responsibilities:**
  - Define the specific ML problem and objectives
  - Manage the project timeline and deliverables
  - Lead the data acquisition and understanding
  - Document the data sources and limitations
  - Help coordinate model comparison

### Member 2: Data Preprocessing & Feature Engineering Lead
- **Responsibilities:**
  - Clean the Komuter dataset (handling missing values, outliers)
  - Create temporal features (peak hours, weekday/weekend, etc.)
  - Engineer network-based features (station popularity, connectivity)
  - Create distance-based features between stations
  - Prepare preprocessed data for modeling

### Member 3: Model Development Lead (Regression Models)
- **Responsibilities:**
  - Implement and tune traditional regression models:
    - Linear Regression
    - Decision Tree Regression
    - Random Forest Regression
    - Gradient Boosting
  - Document model assumptions and limitations
  - Evaluate models using appropriate metrics (RMSE, MAE, R²)

### Member 4: Advanced Model Development Lead
- **Responsibilities:**
  - Implement and tune more advanced models:
    - Neural Networks (MLP Regressor)
    - Support Vector Regression
    - XGBoost/LightGBM
    - Auto-sklearn implementation (required by project spec)
  - Feature importance analysis
  - Hyperparameter optimization
  - Implement time series models (ARIMA, Prophet) for temporal components

### Member 5: Evaluation & Presentation Lead
- **Responsibilities:**
  - Design comprehensive evaluation framework
  - Compare all models using consistent metrics
  - Create visualizations of model performance
  - Develop anomaly detection visualization dashboards
  - Prepare the final presentation slides
  - Coordinate the 5-minute video presentation
  - Ensure all deliverables meet submission requirements

## Collaborative Tasks (All Members)

1. **Initial Brainstorming**
   - Define the specific prediction task
   - Identify potential features and approaches

2. **Model Selection Meeting**
   - Decide on the final set of 5+ models to implement
   - Agree on evaluation metrics

3. **Final Review**
   - Review all components before submission
   - Practice presentation to ensure it fits within 5 minutes

## Handling Large Datasets

### Technical Approaches for Large Data Processing

#### 1. Efficient Data Loading
- **Chunking**: Using `pandas.read_csv()` with the `chunksize` parameter to read large CSV files in manageable chunks
- **Memory-mapped files**: Using NumPy's `memmap` for array-like data structures
- **Data sampling**: Using representative samples during development phase

```python
# Example of chunked processing
chunk_size = 5000
for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
    # Process each chunk
    processed_chunk = process_data(chunk)
    # Append results or aggregate statistics
```

#### 2. Data Preprocessing Optimization
- **Feature selection**: Removing irrelevant features early in the pipeline
- **Dimensionality reduction**: Using PCA, t-SNE, or UMAP for reducing data dimensions
- **Data compression**: Converting high-precision numerical values to lower precision when appropriate
- **Efficient categorical encoding**: Using memory-efficient methods like target encoding

#### 3. Model Training with Large Data
- **Online learning**: Updating model parameters incrementally with batches of data
- **Out-of-core learning**: Using algorithms designed for datasets that don't fit in memory
- **Distributed processing**: Utilizing libraries like Dask or PySpark for parallel computing
- **GPU acceleration**: Leveraging GPU computing for applicable algorithms

#### 4. Scalable Validation Strategies
- **K-fold cross-validation with sampling**: Implementing cross-validation on sampled data
- **Progressive validation**: Evaluating performance incrementally as training proceeds
- **Holdout validation**: Using separate, representative test sets

## Project Timeline
- **Week 6:** ✅ Problem definition, data exploration
- **Weeks 6-7:** ✅ Data preprocessing, feature engineering
- **Weeks 7-9:** 🔄 Model development and tuning (IN PROGRESS)
- **Weeks 10-11:** Model evaluation and comparison, presentation preparation
- **Week 12:** Final submission
- **Week 14:** Final pitching (if selected as finalist)

## Current Project Structure
```
Project_Workflow.txt
README.md
requirements.txt
WIA1006 Group Assignment 2024_25.pdf
data/
    processed/
        komuter_features.csv
        komuter_processed.csv
        komuter_test.csv
        komuter_train.csv
    raw/
        crops_district_production.csv
        komuter_2025.csv
notebooks/
    01_data_exploration.ipynb
    02_data_preprocessing.ipynb
    03_feature_engineering.ipynb
    04_model_development.ipynb
    05_model_evaluation.ipynb
src/
    Introduction.py
    data/
        data_loading.py
        make_dataset.py
    models/
```

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation
1. Clone this repository
2. Create a virtual environment (recommended)
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebooks
Navigate to the notebooks directory and start with `01_data_exploration.ipynb` to understand the workflow:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```