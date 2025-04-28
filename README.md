# WIA1006 Machine Learning Assignment

## Project Overview
This repository contains our group's machine learning project for the WIA1006 Machine Learning course at FCSIT UM. Our goal is to apply machine learning techniques to solve real-world problems using public datasets from the [Malaysian government](https://data.gov.my/). 

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
- **Weeks 6:** ✅ Problem definition, data exploration
- **Week 6-7:** 🔄 Data preprocessing, feature engineering (IN PROGRESS)
- **Weeks 7-9:** Model development and tuning
- **Weeks 10-11:** Model evaluation and comparison, presentation preparation
- **Week 12:** Final submission
- **Week 14:** Final pitching (if selected as finalist)


## Repository Structure
```
├── data/                   # Data storage (raw, processed)
├── notebooks/              # Jupyter notebooks for exploration and modeling
├── src/                    # Source code for utilities and reusable functions
├── models/                 # Saved model files
├── reports/                # Generated analysis reports
└── ML_Project_Notebook.ipynb  # Main project notebook
```
```
komuter_ridership_analysis/
├── data/
│   ├── raw/                  # Original, immutable data
│   │   └── komuter_2025.csv
│   ├── processed/            # Cleaned, transformed data
│   └── external/             # External data sources if needed
├── notebooks/               # Jupyter notebooks for exploration and presentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_model_evaluation.ipynb
├── src/                     # Source code for use in this project
│   ├── __init__.py
│   ├── data/                # Scripts to process data
│   │   ├── __init__.py
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── features/           # Scripts to turn processed data into features
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/             # Scripts to train and evaluate models
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/      # Scripts for creating exploratory and results visualizations
│       ├── __init__.py
│       └── visualize.py
├── reports/                # Generated analysis reports
│   └── figures/            # Generated graphics and figures
├── models/                 # Trained and serialized models
├── environment.yml         # Conda environment file
└── README.md               # Project description and setup instructions