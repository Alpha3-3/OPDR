import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from tqdm import tqdm


# Load pre-trained vectors
def load_vectors(file_path):
    return np.load(file_path)


# DW-PMAD calculation
# Optimized DW-PMAD calculation
def dw_pmad_b(w, X, b):
    w = w / np.linalg.norm(w)  # Normalize direction vector
    projections = X @ w
    abs_diffs = pdist(projections.reshape(-1, 1))  # Efficient pairwise differences

    # Compute top_b_count with bounds check
    num_pairs = len(abs_diffs)
    top_b_count = min(num_pairs - 1, max(1, int((b / 100) * num_pairs)))

    return -np.mean(np.partition(abs_diffs, top_b_count)[:top_b_count])  # Partial sort


    # Orthogonality constraint
def orthogonality_constraint(w, prev_ws, alpha):
    prev_ws = np.array(prev_ws)  # Convert to NumPy array
    if prev_ws.size == 0:
        return 0
    dot_products = np.dot(prev_ws, w) ** 2  # Efficient vectorized computation
    return np.sum(dot_products)



# Optimized DW-PMAD
def dw_pmad(X, b, alpha, target_dim):
    X_centered = X - np.mean(X, axis=0)
    prev_ws = []
    optimal_ws = np.zeros((X.shape[1], target_dim))

    for axis in range(target_dim):
        def constrained_dw_pmad(w):
            return dw_pmad_b(w, X_centered, b) + alpha * orthogonality_constraint(w, prev_ws, alpha)

        result = minimize(constrained_dw_pmad, np.random.randn(X.shape[1]), method='L-BFGS-B')
        optimal_w = result.x / np.linalg.norm(result.x)

        prev_ws.append(optimal_w)
        optimal_ws[:, axis] = optimal_w  # Store directly in NumPy array

    return X_centered @ optimal_ws, optimal_ws


# Project test data using stored DW-PMAD axes
def project_dw_pmad(X, projection_axes):
    return (X - np.mean(X, axis=0)) @ projection_axes


# Accuracy calculation
def calculate_accuracy(original_data, reduced_data, new_original_data, new_reduced_data, k):
    nbrs_original = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(reduced_data)

    original_neighbors = nbrs_original.kneighbors(new_original_data, return_distance=False)
    reduced_neighbors = nbrs_reduced.kneighbors(new_reduced_data, return_distance=False)

    total_matches = np.sum([len(set(original_neighbors[i]) & set(reduced_neighbors[i])) for i in range(len(new_original_data))])

    return total_matches / (len(new_original_data) * k)


# Worker function for multiprocessing
def process_parameters(params):
    dim, target_ratio, b, alpha = params  # Unpack tuple here
    target_dim = max(1, int(dim * target_ratio))
    print(f"Processing: Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}, Target Dim={target_dim}")

    X_train = training_vectors[:, :dim]
    X_test = testing_vectors[:, :dim]

    # Compute DW-PMAD
    print(f"Starting DW-PMAD for Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}")
    X_dw_pmad, dw_pmad_axes = dw_pmad(X_train, b, alpha, target_dim)
    print(f"DW-PMAD complete for Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}")

    # Compute PCA
    print(f"Starting PCA for Dimension={dim}, Target Ratio={target_ratio}")
    pca = PCA(n_components=target_dim).fit(X_train)
    X_pca = pca.transform(X_train)
    print(f"PCA complete for Dimension={dim}, Target Ratio={target_ratio}")

    # Transform test data
    new_dw_pmad = project_dw_pmad(X_test, dw_pmad_axes)
    new_pca = pca.transform(X_test)

    results = []  # Collect results in a local list
    for k in k_values:
        acc_dw_pmad = calculate_accuracy(X_train, X_dw_pmad, X_test, new_dw_pmad, k)
        acc_pca = calculate_accuracy(X_train, X_pca, X_test, new_pca, k)

        better_method = 'dw_pmad' if acc_dw_pmad > acc_pca else 'pca'
        print(f"Results for dim={dim}, target_ratio={target_ratio}, b={b}, alpha={alpha}, k={k}: DW-PMAD Accuracy={acc_dw_pmad}, PCA Accuracy={acc_pca}, Better Method={better_method}")

        results.append([dim, target_ratio, b, alpha, k, acc_dw_pmad, acc_pca, better_method])

    return results  # Return the results instead of appending to a shared list


# Parameter settings
b_values = [50,70,85, 100]
k_values = [3,6, 10,15]
alpha_values = [1,5,10,25]
dimensions = [10] # 299
target_dims = [0.2,0.4, 0.6, 0.8]

# Load data
training_vectors = load_vectors('training_vectors_600.npy')
testing_vectors = load_vectors('testing_vectors_1000.npy')

# Generate all unique parameter combinations
param_combinations = list(itertools.product(dimensions, target_dims, b_values, alpha_values))

# Use multiprocessing to parallelize
if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        results_list = pool.map(process_parameters, param_combinations)  # Use `map`, not `starmap`

    # Flatten list of lists
    flat_results = [item for sublist in results_list if sublist is not None for item in sublist]

    # Convert to DataFrame and export
    results_df = pd.DataFrame(flat_results, columns=['Dimension', 'Target Ratio', 'b', 'alpha', 'k', 'DW-PMAD Accuracy', 'PCA Accuracy', 'Better Method'])
    results_df.to_csv('parameter_sweep_results_fasttext1.csv', index=False)
    print(results_df)
