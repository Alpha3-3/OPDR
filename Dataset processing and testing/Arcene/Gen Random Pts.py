import numpy as np
import random
import os
from tqdm import tqdm

# List of file paths to include
file_paths = [
    'ARCENE/arcene_train.data',
    'ARCENE/arcene_valid.data',
    'ARCENE/arcene_test.data'
]

# Initialize an empty list to store the vectors from all files
vectors = []

# Iterate over each file path and load vectors
for file_path in file_paths:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)  # Skip the header line if it contains metadata
        for line in tqdm(file, desc=f"Reading vectors from {os.path.basename(file_path)}"):
            # Use whitespace as delimiter (change to split(',') if needed)
            parts = line.strip().split()
            # Append the vector (ignoring the first element if it is an ID/label)
            vectors.append(np.array(parts[1:], dtype=float))

# Convert the list of vectors to a NumPy array
vectors = np.array(vectors)

# Define sample sizes for training and testing
num_training = 600
num_testing = 297

# Check that there are enough vectors for non-overlapping samples
total_required = num_training + num_testing
if len(vectors) < total_required:
    raise ValueError(
        f"Not enough vectors for non-overlapping training and testing samples. "
        f"Total required: {total_required}, available: {len(vectors)}."
    )

# Create a list of all indices
all_indices = list(range(len(vectors)))

# --- Sample Training Vectors ---
random.seed(1)
np.random.seed(1)
training_indices = random.sample(all_indices, num_training)
training_vectors = vectors[training_indices]

# Save training vectors
np.save('training_vectors_600_Arcene.npy', training_vectors)
print("Training vectors have been selected and saved successfully.")

# --- Sample Testing Vectors from the remaining indices ---
# Remove the training indices from the pool
remaining_indices = list(set(all_indices) - set(training_indices))
if len(remaining_indices) < num_testing:
    raise ValueError(
        f"Not enough remaining vectors for testing. "
        f"Required: {num_testing}, available: {len(remaining_indices)}."
    )

random.seed(2)
np.random.seed(2)
testing_indices = random.sample(remaining_indices, num_testing)
testing_vectors = vectors[testing_indices]

# Save testing vectors
np.save('testing_vectors_297_Arcene.npy', testing_vectors)
print("Testing vectors have been selected and saved successfully.")
