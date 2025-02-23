import numpy as np
import random
import os
from tqdm import tqdm

# Set random seeds for reproducibility
random.seed(1)
np.random.seed(1)

# Path to the CSV file
csv_file_path = r'D:\My notes\UW\HPDIC Lab\OPDR\datasets\HIGGS.csv'
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"The file {csv_file_path} does not exist")

# First, count the total number of lines to set the tqdm total
with open(csv_file_path, 'r') as f:
    total_lines = sum(1 for _ in f)

# Load the CSV data line-by-line with a progress bar
data_list = []
with open(csv_file_path, 'r') as f:
    # If your CSV file contains a header, uncomment the next line:
    # next(f)
    for line in tqdm(f, total=total_lines, desc="Reading CSV"):
        # Split by comma and convert each value to float.
        # Adjust the conversion if your CSV contains non-numeric data.
        values = line.strip().split(',')
        data_list.append([float(v) for v in values])

# Convert the list to a NumPy array
data = np.array(data_list)

# Number of rows (points) to randomly select
num_points = 300
if data.shape[0] < num_points:
    raise ValueError("The CSV file does not contain enough rows.")

# Randomly select indices
selected_indices = random.sample(range(data.shape[0]), num_points)

# Use tqdm to show progress while collecting the selected points
selected_points = []
for idx in tqdm(selected_indices, desc="Selecting random points"):
    selected_points.append(data[idx])
selected_points = np.array(selected_points)

# Save the selected points to a .npy file
output_file = 'training_vectors_300.npy'
np.save(output_file, selected_points)

print("Selected random points have been saved successfully to", output_file)

# Number of rows (points) to randomly select
num_points = 1000
if data.shape[0] < num_points:
    raise ValueError("The CSV file does not contain enough rows.")

# Randomly select indices
selected_indices = random.sample(range(data.shape[0]), num_points)

# Use tqdm to show progress while collecting the selected points
selected_points = []
for idx in tqdm(selected_indices, desc="Selecting random points"):
    selected_points.append(data[idx])
selected_points = np.array(selected_points)

# Save the selected points to a .npy file
output_file = 'testing_vectors_1000.npy'
np.save(output_file, selected_points)

print("Selected random points have been saved successfully to", output_file)
