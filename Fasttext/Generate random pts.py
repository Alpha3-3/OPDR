import numpy as np
import random
import os
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(2)  # Any fixed integer value
np.random.seed(2)  # Ensures consistency across numpy operations

# Path to the .vec file
vec_file_path = r'D:\My notes\UW\HPDIC Lab\OPDR\datasets\wiki-news-300d-1M.vec'

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
    # Process each line after the header
    for line in tqdm(file, desc="Reading vectors"):
        parts = line.strip().split()
        # Convert vector parts to float and append to the list
        vectors.append(np.array(parts[1:], dtype=float))

# Convert list to a NumPy array
vectors = np.array(vectors)

# Check if there are enough vectors to sample from
if len(vectors) < num_points:
    raise ValueError("The file contains fewer vectors than the number of points requested.")

# Randomly select num_points vectors with reproducibility
selected_indices = random.sample(range(len(vectors)), num_points)
selected_vectors = vectors[selected_indices]

# Save selected vectors to a .npy file
npy_file_path = 'testing_vectors_600_Fasttext.npy'
np.save(npy_file_path, selected_vectors)

print("Vectors have been selected and saved successfully.")
