import numpy as np
import pandas as pd

# Load pre-trained vectors
def load_vectors(file_path):
    return np.load(file_path)

training_vectors = load_vectors('training_vectors_600_Dorothea.npy')
testing_vectors = load_vectors('testing_vectors_600_Dorothea.npy')

# For inspection, only use the first 3000 dimensions
training_inspect = training_vectors[:, :3000]
testing_inspect = testing_vectors[:, :3000]

# Print statistics of the vectors (first 3000 dimensions)
print("Training Vectors Description (first 3000 dimensions):")
print(pd.DataFrame(training_inspect).describe())
print(pd.DataFrame(training_inspect).head().to_string())

print("\nTesting Vectors Description (first 3000 dimensions):")
print(pd.DataFrame(testing_inspect).describe())
print(pd.DataFrame(testing_inspect).head().to_string())

# Check for overlaps by comparing the full training and testing vectors
# We will round them to avoid precision issues when comparing
rounded_training_vectors = np.round(training_vectors, decimals=5)
rounded_testing_vectors = np.round(testing_vectors, decimals=5)

# Use np.isin to check if any element from testing_vectors exists in training_vectors
overlap_found = np.isin(rounded_testing_vectors.tobytes(), rounded_training_vectors.tobytes())

# Check if there's any overlap
if np.any(overlap_found):
    print("\nOverlap found between training and testing vectors.")
else:
    print("\nNo overlap found between training and testing vectors.")
