import numpy as np
import pandas as pd

# Load pre-trained vectors
def load_vectors(file_path):
    return np.load(file_path)

training_vectors = load_vectors('training_vectors_300_CIFAR-10.npy')
testing_vectors = load_vectors('testing_vectors_300_CIFAR-10.npy')

# Print statistics of the vectors
print("Training Vectors Description:")
print(pd.DataFrame(training_vectors).describe())
print(pd.DataFrame(training_vectors).head().to_string())

print("\nTesting Vectors Description:")
print(pd.DataFrame(testing_vectors).describe())
print(pd.DataFrame(testing_vectors).head().to_string())

# Check for overlaps by comparing the training and testing vectors
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
