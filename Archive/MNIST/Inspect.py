import numpy as np
import pandas as pd

# Load pre-trained vectors
def load_vectors(file_path):
    return np.load(file_path)

training_vectors = load_vectors('training_vectors_600_MNIST.npy')
testing_vectors = load_vectors('testing_vectors_600_MNIST.npy')

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

# Count how many features (dimensions) in the first 3000 columns are not fully zero.
nonzero_training_features = np.sum(np.any(training_inspect != 0, axis=0))
nonzero_testing_features = np.sum(np.any(testing_inspect != 0, axis=0))

print("\nNonzero feature counts (first 3000 dimensions):")
print(f"Training vectors: {nonzero_training_features} out of {training_inspect.shape[1]} dimensions are not fully zero.")
print(f"Testing vectors: {nonzero_testing_features} out of {testing_inspect.shape[1]} dimensions are not fully zero.")

# Updated Overlap Check:
# Round the full vectors to avoid precision issues when comparing
rounded_training_vectors = np.round(training_vectors, decimals=5)
rounded_testing_vectors = np.round(testing_vectors, decimals=5)

# Convert each vector (row) into a tuple and form sets for comparison
training_set = set(map(tuple, rounded_training_vectors))
testing_set = set(map(tuple, rounded_testing_vectors))

# Find the overlapping vectors using set intersection
overlap = training_set.intersection(testing_set)
print(f"\nNumber of overlapping vectors: {len(overlap)}")
if len(overlap) > 0:
    print("Overlap found between training and testing vectors.")
else:
    print("No overlap found between training and testing vectors.")
