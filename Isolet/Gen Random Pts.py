import numpy as np
import random
import os
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(1)  # Any fixed integer value
np.random.seed(1)  # Ensures consistency across numpy operations

# Path to the .vec file
vec_file_path = 'isolet dataset/isolet1+2+3+4.data'

# Number of points to select
num_points = 600

# Ensure the file exists
if not os.path.exists(vec_file_path):
    raise FileNotFoundError(f"The file {vec_file_path} does not exist")

# Initialize an empty list to store the vectors
vectors = []

# Reading the .vec file, skipping the first line since it contains metadata
with open(vec_file_path, 'r', encoding='utf-8') as file:
    next(file)  # Skip the header line
    for line in tqdm(file, desc="Reading vectors"):
        parts = line.strip().split(',')  # Use ',' instead of space
        vectors.append(np.array(parts[1:], dtype=float))  # Convert to float

# Convert list to a NumPy array
vectors = np.array(vectors)

# Check if there are enough vectors to sample from
if len(vectors) < num_points:
    raise ValueError("The file contains fewer vectors than the number of points requested.")

# Randomly select num_points vectors with reproducibility
selected_indices = random.sample(range(len(vectors)), num_points)
selected_vectors = vectors[selected_indices]

# Save selected vectors to a .npy file
npy_file_path = 'training_vectors_600_Isolet.npy'
np.save(npy_file_path, selected_vectors)

print("Vectors have been selected and saved successfully.")
