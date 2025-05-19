import scanpy as sc
import numpy as np

# Load the processed PBMC3k dataset
adata = sc.datasets.pbmc3k_processed()

# Get the expression matrix (adata.X). If it's stored as a sparse matrix, convert it to dense.
if not isinstance(adata.X, np.ndarray):
    X = adata.X.toarray()
else:
    X = adata.X

# Filter rows: keep only those vectors (rows) where ALL dimensions are valid (not NaN)
valid_mask = np.all(~np.isnan(X), axis=1)
X_valid = X[valid_mask]
print(f"Total number of valid vectors available: {X_valid.shape[0]}")
print(f"Dimension of each vector: {X_valid.shape[1]}")

# Set a random seed for reproducibility for the entire train/test split
random_seed = 1 # Changed seed for a combined operation
np.random.seed(random_seed)

# Define the number of vectors you want for training and testing
num_training_vectors = 1200  # You can change this value as needed
num_testing_vectors = 300   # You can change this value as needed

# Check if the requested number of vectors exceeds the available data
if num_training_vectors + num_testing_vectors > X_valid.shape[0]:
    raise ValueError(f"The sum of training ({num_training_vectors}) and testing ({num_testing_vectors}) vectors "
                     f"exceeds the total number of available valid vectors ({X_valid.shape[0]}).")

# Generate a permutation of all available indices
all_indices = np.arange(X_valid.shape[0])
shuffled_indices = np.random.permutation(all_indices)

# Select indices for the training set
training_indices = shuffled_indices[:num_training_vectors]

# Select indices for the testing set (ensuring no overlap)
testing_indices = shuffled_indices[num_training_vectors : num_training_vectors + num_testing_vectors]

# Retrieve the training vectors
training_vectors = X_valid[training_indices]

# Save the training vectors to a .npy file
np.save("training_vectors_1200_PBMC3k.npy", training_vectors)
print(f"\nSaved {training_vectors.shape[0]} training vectors with dimension {training_vectors.shape[1]} to 'training_vectors_1200_PBMC3k.npy'.")
# print("Sample of training vectors:")
# print(training_vectors[:5]) # Print first 5 for brevity

# Retrieve the testing vectors
testing_vectors = X_valid[testing_indices]

# Save the testing vectors to a .npy file
np.save("testing_vectors_300_PBMC3k.npy", testing_vectors)
print(f"Saved {testing_vectors.shape[0]} testing vectors with dimension {testing_vectors.shape[1]} to 'testing_vectors_300_PBMC3k.npy'.")
# print("Sample of testing vectors:")
# print(testing_vectors[:5]) # Print first 5 for brevity

# --- Verification of no overlap (optional) ---
# Convert rows to a set of tuples for easy comparison
set_training_vectors = set(map(tuple, training_vectors))
set_testing_vectors = set(map(tuple, testing_vectors))

overlap = set_training_vectors.intersection(set_testing_vectors)
if not overlap:
    print("\nVerification successful: No overlap between training and testing sets.")
else:
    print(f"\nVerification FAILED: Found {len(overlap)} overlapping vectors between training and testing sets.")

print(f"\nTotal vectors in training set: {len(training_vectors)}")
print(f"Total vectors in testing set: {len(testing_vectors)}")
print(f"Total unique vectors across both sets: {len(set_training_vectors.union(set_testing_vectors))}")