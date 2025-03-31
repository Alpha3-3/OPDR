import numpy as np
import random
import os
from tqdm import tqdm

# Path to the .vec file
vec_file_path = 'DOROTHEA/dorothea_train.data'

# Ensure the file exists
if not os.path.exists(vec_file_path):
    raise FileNotFoundError(f"The file {vec_file_path} does not exist")

# Read the entire file into memory
with open(vec_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Read and process the header (if present)
header = lines[0].strip()
if ',' in header:
    header_parts = header.split(',')
else:
    header_parts = header.split()

# Instead of using the header's n_features, compute it from the data
max_index = 0
data_lines = lines[1:]
for line in data_lines:
    tokens = line.strip().split(',') if ',' in line else line.strip().split()
    for token in tokens:
        if token:  # skip empty tokens
            idx = int(token)
            if idx > max_index:
                max_index = idx

# Set the number of features as the maximum index found
n_features = max_index

# Now, process each data line to create a dense vector
vectors = []
for line in tqdm(data_lines, desc="Reading vectors"):
    tokens = line.strip().split(',') if ',' in line else line.strip().split()
    # Initialize a zero vector of size n_features
    vec = np.zeros(n_features, dtype=float)
    for token in tokens:
        if token:  # skip empty tokens
            # Assuming the file uses 1-indexing for feature indices
            idx = int(token) - 1
            vec[idx] = 1.0
    vectors.append(vec)

# Convert list to a NumPy array (all vectors now have the same shape)
vectors = np.array(vectors)

# Number of points to select
num_points = 600
if len(vectors) < num_points:
    raise ValueError("The file contains fewer vectors than the number of points requested.")

# Set random seed for reproducibility and randomly sample num_points vectors
random.seed(1)
np.random.seed(1)
selected_indices = random.sample(range(len(vectors)), num_points)
selected_vectors = vectors[selected_indices]

# Save the selected vectors to a .npy file
np.save('training_vectors_600_Dorothea.npy', selected_vectors)
print("Vectors have been selected and saved successfully.")

# Set random seed for reproducibility and randomly sample num_points vectors
random.seed(2)
np.random.seed(2)
selected_indices = random.sample(range(len(vectors)), num_points)
selected_vectors = vectors[selected_indices]

# Save the selected vectors to a .npy file
np.save('testing_vectors_600_Dorothea.npy', selected_vectors)
print("Vectors have been selected and saved successfully.")