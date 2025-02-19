import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import time

# --- DW-PMAD Functions ---

def dw_pmad_b(w, X, b):
    """
    Compute the negative mean of the top-b percentage of pairwise differences
    along the projection defined by w.
    """
    w = w / np.linalg.norm(w)  # Normalize the direction vector
    projections = X @ w        # Project all data points onto w
    # Compute all pairwise differences (quadratic cost)
    abs_diffs = pdist(projections.reshape(-1, 1))

    num_pairs = len(abs_diffs)
    # Determine how many of the smallest differences to consider
    top_b_count = min(num_pairs - 1, max(1, int((b / 100) * num_pairs)))

    # Use a partial sort to get the top b differences
    return -np.mean(np.partition(abs_diffs, top_b_count)[:top_b_count])

def orthogonality_constraint(w, prev_ws):
    """
    Compute the penalty for non-orthogonality with previously computed axes.
    """
    return sum((np.dot(w, prev_w) ** 2) for prev_w in prev_ws)

def dw_pmad(X, b, alpha, target_dim):
    """
    Compute the DW-PMAD projection of the data onto target_dim axes.
    Each axis is computed by optimizing the dw_pmad_b objective while ensuring
    orthogonality with previously found axes.
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    prev_ws, optimal_ws = [], []

    for axis in range(target_dim):
        # The objective function to minimize for the current axis
        def constrained_dw_pmad(w):
            return dw_pmad_b(w, X_centered, b) + alpha * orthogonality_constraint(w, prev_ws)

        # Start optimization from a random initial vector
        result = minimize(constrained_dw_pmad, np.random.randn(X.shape[1]), method='L-BFGS-B')
        optimal_w = result.x / np.linalg.norm(result.x)

        prev_ws.append(optimal_w)
        optimal_ws.append(optimal_w)

    # Return the projected data and the axes
    return X_centered @ np.column_stack(optimal_ws), np.column_stack(optimal_ws)

# --- PCA and Timing Illustration ---

# Create synthetic data
np.random.seed(1)
n_samples = 600  # Number of samples (n)
n_features = 299  # Dimensionality (d)
X = np.random.randn(n_samples, n_features)

# Parameters for DW-PMAD
b = 70          # Percentage for selecting top pairwise differences
alpha = 5       # Weight for the orthogonality penalty
target_dim = 200  # Number of dimensions to reduce to

# Time PCA (using sklearn's optimized routines)
start_pca = time.time()
pca = PCA(n_components=target_dim)
X_pca = pca.fit_transform(X)
end_pca = time.time()
pca_time = end_pca - start_pca
print(f"PCA completed in {pca_time:.4f} seconds.")

# Time DW-PMAD (which uses iterative optimization)
start_dwp = time.time()
X_dw_pmad, dw_axes = dw_pmad(X, b, alpha, target_dim)
end_dwp = time.time()
dwp_time = end_dwp - start_dwp
print(f"DW-PMAD completed in {dwp_time:.4f} seconds.")

# Show the shape of the transformed data
print(f"Shape after PCA: {X_pca.shape}")
print(f"Shape after DW-PMAD: {X_dw_pmad.shape}")
