# WIA1006 Machine Learning Assignment

## Project Overview
This repository contains our group's machine learning project for the WIA1006 Machine Learning course at FCSIT UM. Our goal is to apply machine learning techniques to solve real-world problems using public datasets from the [Malaysian government](https://data.gov.my/). 

## Theme Selection
We have selected the following dataset for our project:

### Hourly Origin-Destination Ridership: Komuter
- **Dataset Source**: [data.gov.my - Ridership OD Komuter](https://data.gov.my/data-catalogue/ridership_od_komuter?)
- **Data Volume**: ~30,000+ records

## Project Objective
We are developing a **Ridership Prediction Model** to predict the number of passengers traveling between any origin-destination pair based on temporal features (hour, day of week) and station characteristics.

This model will help KTM optimize train frequencies and capacities based on predicted demand, improve resource allocation during peak hours, and enable better planning for maintenance schedules during low-ridership periods.

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

### Member 5: Evaluation & Presentation Lead
- **Responsibilities:**
  - Design comprehensive evaluation framework
  - Compare all models using consistent metrics
  - Create visualizations of model performance
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

## Project Purpose 

### Theme: Hourly Origin-Destination Ridership (Komuter)

#### Potential Project Purposes:
1. **Predictive Ridership Modeling**
   - Forecast passenger volumes based on time, day, season, and other factors
   - Help transportation authorities optimize train schedules and capacity
   - Enable dynamic resource allocation during peak and off-peak hours

2. **Travel Pattern Analysis**
   - Identify common commuter routes and patterns
   - Analyze how travel behaviors change over time (daily, weekly, seasonally)
   - Discover underserved routes or overburdened sections

3. **Anomaly Detection System**
   - Develop models to detect unusual ridership patterns
   - Create early warning systems for service disruptions
   - Identify special events that significantly impact the transportation network

4. **Route Optimization Framework**
   - Build models to recommend optimal train frequencies and capacities
   - Suggest new routes or modifications to existing routes
   - Minimize travel time and maximize passenger satisfaction

### Recommended ML Objectives

#### 1. Ridership Prediction Model
**Objective:** Develop a regression model to predict the number of passengers traveling between any origin-destination pair based on temporal features (hour, day of week) and station characteristics.

**Business Value:**
Help KTM optimize train frequencies and capacities based on predicted demand
Improve resource allocation during peak hours
Enable better planning for maintenance schedules during low-ridership periods

#### 2. Peak Period Classification
**Objective:** Build a classification model to predict whether a specific time period will experience high, medium, or low ridership volumes across the network.

**Business Value:**
Allow for dynamic staffing and resource allocation
Help passengers plan their journeys to avoid crowded periods
Support pricing strategies like peak/off-peak fares

#### 3. Station Clustering for Service Planning
**Objective:** Develop a clustering model to group stations with similar ridership patterns and characteristics.

**Business Value:**
Identify stations that may benefit from similar service improvements
Discover natural service zones for planning purposes
Support targeted marketing campaigns to specific station clusters

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
```

## Setup Instructions
```bash
# Clone the repository
git clone [repository-url]

# Navigate to the project directory
cd [project-directory]

# Install required packages
pip install -r requirements.txt
```

## Contact
For any questions regarding this project, please contact:
- [Group Representative Name] - [Email]

---
*Last Updated: April 23, 2025*