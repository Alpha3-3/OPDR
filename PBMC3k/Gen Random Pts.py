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
print(X_valid.shape)
# Set a random seed for reproducibility
random_seed = 1
np.random.seed(random_seed)

# Define the number of vectors you want to sample
num_vectors = 600  # You can change this value as needed

# Ensure we don't try to sample more vectors than available in X_valid
num_vectors = min(num_vectors, X_valid.shape[0])

# Randomly select row indices without replacement
sample_indices = np.random.choice(X_valid.shape[0], size=num_vectors, replace=False)

# Retrieve the sampled vectors
sampled_vectors = X_valid[sample_indices]

# Save the sampled vectors to a .npy file
np.save("training_vectors_600_PBMC3k.npy", sampled_vectors)

print(f"Saved {sampled_vectors.shape[0]} vectors with dimension {sampled_vectors.shape[1]} to 'training_vectors_600_PBMC3k.npy'.")
print(sampled_vectors)


# Set a random seed for reproducibility
random_seed = 2
np.random.seed(random_seed)

# Define the number of vectors you want to sample
num_vectors = 600  # You can change this value as needed

# Ensure we don't try to sample more vectors than available in X_valid
num_vectors = min(num_vectors, X_valid.shape[0])

# Randomly select row indices without replacement
sample_indices = np.random.choice(X_valid.shape[0], size=num_vectors, replace=False)

# Retrieve the sampled vectors
sampled_vectors = X_valid[sample_indices]

# Save the sampled vectors to a .npy file
np.save("testing_vectors_600_PBMC3k.npy", sampled_vectors)

print(f"Saved {sampled_vectors.shape[0]} vectors with dimension {sampled_vectors.shape[1]} to 'testing_vectors_600_PBMC3k.npy'.")
print(sampled_vectors)