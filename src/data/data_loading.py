# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
sns.set_context("notebook", font_scale=1.2)

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'C:/Users/User/Documents/UM Academics/SEM 2/WIA1006-MACHINE LEARNING/Assignment/data/raw/komuter_2025.csv'
df = pd.read_csv(file_path)

# Check the first few rows of the data
print(f"Dataset shape: {df.shape}")
df.head()
df.info()