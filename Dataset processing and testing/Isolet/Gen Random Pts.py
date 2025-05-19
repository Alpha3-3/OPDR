import numpy as np
import random
import os
from tqdm import tqdm

# Set random seed for reproducibility for the entire train/test split
# It's good practice to set both if you are using both 'random' and 'numpy.random'
# For this specific script, np.random.permutation will be the key random operation for splitting.
fixed_seed = 1
random.seed(fixed_seed)
np.random.seed(fixed_seed)

# Path to the .vec file
vec_file_path = 'isolet dataset/isolet1+2+3+4.data' # Ensure this path is correct

# Define the number of points you want for training and testing
num_training_points = 1200  # You can change this value as needed
num_testing_points = 300   # You can change this value as needed

# Ensure the file exists
if not os.path.exists(vec_file_path):
    raise FileNotFoundError(f"The file {vec_file_path} does not exist. Please check the path.")

# Initialize an empty list to store the vectors
all_vectors_list = []

# Reading the .vec file
print(f"Reading vectors from {vec_file_path}...")
try:
    with open(vec_file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(tqdm(file, desc="Processing lines")):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split(',')
            # Assuming the first part is an identifier/label and the rest are features.
            # If all parts are features, adjust accordingly.
            # For Isolet, it seems the last column might be the class, but the original code
            # treated all but the first (if any) as features.
            # Let's assume all comma-separated values are part of the feature vector.
            # If the first column is an ID or not a feature, use parts[1:]
            try:
                # Attempt to convert all parts to float. Adjust if some columns are not numeric.
                # The original code used parts[1:], implying the first column might be different or an ID.
                # For Isolet data, often the last column is the class label.
                # Let's assume for now all parts are features, or you need to adjust data extraction.
                # Based on original: parts[1:]
                # If all columns are data:
                # feature_vector = np.array(parts, dtype=float)

                # The original code snippet `vectors.append(np.array(parts[1:], dtype=float))`
                # suggests the first element of `parts` might not be part of the feature vector,
                # or the file has a specific structure not fully detailed.
                # However, the problem description focuses on splitting vectors, not their interpretation.
                # Let's proceed assuming each line gives one vector.
                # A common format for Isolet is that data points are rows of features,
                # sometimes with a class label at the end.
                # The original code skipped a header line, but the provided snippet did not.
                # I'll assume each line is a data point.
                # Example from common Isolet datasets: feature1,feature2,...,featureN,classLabel
                # If this is the case, and you only want features:
                # all_vectors_list.append(np.array(parts[:-1], dtype=float))
                # If the entire line (once split) is the vector:
                all_vectors_list.append(np.array(parts, dtype=float)) # Assuming all parts are features

            except ValueError as e:
                print(f"Warning: Could not convert line {line_number + 1} to float: '{line}'. Error: {e}. Skipping.")
                continue
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    raise

if not all_vectors_list:
    raise ValueError("No valid vectors were read from the file. Please check the file content and format.")

# Convert list to a NumPy array
all_vectors = np.array(all_vectors_list)

print(f"\nSuccessfully read {all_vectors.shape[0]} vectors.")
if all_vectors.shape[0] > 0:
    print(f"Dimension of each vector: {all_vectors.shape[1]}")
else:
    raise ValueError("Vector array is empty after processing the file.")


# Check if the requested number of vectors exceeds the available data
total_requested_points = num_training_points + num_testing_points
if total_requested_points > all_vectors.shape[0]:
    raise ValueError(f"The sum of training ({num_training_points}) and testing ({num_testing_points}) points ({total_requested_points}) "
                     f"exceeds the total number of available vectors ({all_vectors.shape[0]}).")

# Generate a permutation of all available indices
all_indices = np.arange(all_vectors.shape[0])
shuffled_indices = np.random.permutation(all_indices)

# Select indices for the training set
training_indices = shuffled_indices[:num_training_points]

# Select indices for the testing set (ensuring no overlap)
testing_indices = shuffled_indices[num_training_points : num_training_points + num_testing_points]

# Retrieve the training vectors
training_vectors = all_vectors[training_indices]

# Save the training vectors to a .npy file
training_npy_file_path = 'training_vectors_1200_Isolet.npy'
np.save(training_npy_file_path, training_vectors)
print(f"\nSaved {training_vectors.shape[0]} training vectors with dimension {training_vectors.shape[1]} to '{training_npy_file_path}'.")

# Retrieve the testing vectors
testing_vectors = all_vectors[testing_indices]

# Save the testing vectors to a .npy file
testing_npy_file_path = 'testing_vectors_300_Isolet.npy'
np.save(testing_npy_file_path, testing_vectors)
print(f"Saved {testing_vectors.shape[0]} testing vectors with dimension {testing_vectors.shape[1]} to '{testing_npy_file_path}'.")

# --- Verification of no overlap (optional) ---
# This can be memory intensive for very large datasets if vectors are large
# For simplicity, we can check if any index is shared (which it shouldn't be by design)
shared_indices = np.intersect1d(training_indices, testing_indices)
if len(shared_indices) == 0:
    print("\nVerification successful: No overlap between training and testing sets based on indices.")
else:
    print(f"\nVerification FAILED: Found {len(shared_indices)} overlapping indices between training and testing sets.")

print(f"\nTotal vectors in training set: {len(training_vectors)}")
print(f"Total vectors in testing set: {len(testing_vectors)}")