# prompt: data preprocessing with numpy

import numpy as np

# Sample data (replace with your actual data)
data = np.array([[1, 2, np.nan],
                 [4, 5, 6],
                 [7, np.nan, 9],
                 [10, 11, 12]])

# 1. Handling Missing Values
# Replace NaN values with the mean of the column
data = np.nan_to_num(data, nan=np.nanmean(data))


# 2. Feature Scaling (Min-Max Scaling)
# Scale features to the range [0, 1]
min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)

scaled_data = (data - min_vals) / (max_vals - min_vals)

# 3. Data Normalization (L2 Normalization)
# Normalize each row to have unit length
norms = np.linalg.norm(scaled_data, axis=1, keepdims=True)
normalized_data = scaled_data / norms


# Print the preprocessed data
print("Original Data:\n", data)
print("\nPreprocessed Data (NaN replaced with column mean, Min-Max scaled, and L2 Normalized):\n", normalized_data)