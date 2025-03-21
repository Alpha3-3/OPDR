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
def dw_pmad_b_modified(w, X, b):
    # Normalize the direction vector
    w = w / np.linalg.norm(w)
    projections = X @ w  # Project the data along w
    N = projections.shape[0]

    # Determine how many neighbors to consider per point (exclude the point itself)
    b_count = max(1, int((b / 100) * (N - 1)))

    # Compute the pairwise absolute differences using broadcasting.
    # This produces an N x N matrix where entry (i,j) = |projection_i - projection_j|
    diff_matrix = np.abs(projections.reshape(-1, 1) - projections.reshape(1, -1))

    # To avoid selecting a point's zero difference with itself, set the diagonal to infinity.
    np.fill_diagonal(diff_matrix, np.inf)

    # For each row (each point), select the b_count smallest differences.
    # np.partition along axis=1 gives the b_count smallest values in an unsorted order.
    local_b_diffs = np.partition(diff_matrix, b_count, axis=1)[:, :b_count]

    # Compute the local mean difference for each point.
    local_means = np.mean(local_b_diffs, axis=1)

    # Return the negative overall mean of these local means.
    return -np.mean(local_means)


# Orthogonality constraint
def orthogonality_constraint(w, prev_ws, alpha):
    return sum((np.dot(w, prev_w) ** 2) for prev_w in prev_ws)

# In this version, X is assumed to be standardized already.
def dw_pmad(X, b, alpha, target_dim):
    prev_ws, optimal_ws = [], []
    for axis in range(target_dim):
        def constrained_dw_pmad(w):
            return dw_pmad_b_modified(w, X, b) + alpha * orthogonality_constraint(w, prev_ws, alpha)
        result = minimize(constrained_dw_pmad, np.random.randn(X.shape[1]), method='L-BFGS-B')
        optimal_w = result.x / np.linalg.norm(result.x)
        prev_ws.append(optimal_w)
        optimal_ws.append(optimal_w)
    # Project the data using the computed axes
    return X @ np.column_stack(optimal_ws), np.column_stack(optimal_ws)

# Project test data using stored DW-PMAD axes.
# (Now X is already standardized, so we do not subtract the mean again.)
def project_dw_pmad(X, projection_axes):
    return X @ projection_axes

# Accuracy calculation remains the same.
def calculate_accuracy(original_data, reduced_data, new_original_data, new_reduced_data, k):
    nbrs_original = NearestNeighbors(n_neighbors=k).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=k).fit(reduced_data)

    total_matches = sum(
        len(set(nbrs_original.kneighbors(new_original_data[i].reshape(1, -1), return_distance=False)[0]) &
            set(nbrs_reduced.kneighbors(new_reduced_data[i].reshape(1, -1), return_distance=False)[0]))
        for i in range(len(new_original_data))
    )
    return total_matches / (len(new_original_data) * k)

# Worker function for multiprocessing
def process_parameters(params, results_list):
    dim, target_ratio, b, alpha = params
    target_dim = max(1, int(dim * target_ratio))
    print(f"Processing: Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}, Target Dim={target_dim}")

    # Set a seed for reproducibility
    np.random.seed(1)

    # Determine the total number of dimensions (columns)
    total_dims = training_vectors.shape[1]
    # Randomly select 'dim' unique indices from the available dimensions
    selected_dims = np.random.choice(total_dims, size=dim, replace=False)
    # Extract the chosen dimensions for both training and testing sets
    X_train = training_vectors[:, selected_dims]
    X_test = testing_vectors[:, selected_dims]

    # --- Standardization Process ---
    # Compute training statistics
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    # Avoid division by zero
    train_std[train_std == 0] = 1
    # Standardize both training and test sets using training data statistics
    X_train_standardized = (X_train - train_mean) / train_std
    X_test_standardized = (X_test - train_mean) / train_std
    # --------------------------------

    print(f"Starting DW-PMAD for Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}")
    X_dw_pmad, dw_pmad_axes = dw_pmad(X_train_standardized, b, alpha, target_dim)
    print(f"DW-PMAD complete for Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}")

    print(f"Starting PCA for Dimension={dim}, Target Ratio={target_ratio}")
    pca = PCA(n_components=target_dim).fit(X_train_standardized)
    X_pca = pca.transform(X_train_standardized)
    print(f"PCA complete for Dimension={dim}, Target Ratio={target_ratio}")

    # Transform test data using the stored DW-PMAD axes and PCA
    new_dw_pmad = project_dw_pmad(X_test_standardized, dw_pmad_axes)
    new_pca = pca.transform(X_test_standardized)

    # Store results for different k-values
    for k in k_values:
        acc_dw_pmad = calculate_accuracy(X_train_standardized, X_dw_pmad, X_test_standardized, new_dw_pmad, k)
        acc_pca = calculate_accuracy(X_train_standardized, X_pca, X_test_standardized, new_pca, k)
        better_method = 'dw_pmad' if acc_dw_pmad > acc_pca else 'pca'
        print(f"Results for dim={dim}, target_ratio={target_ratio}, b={b}, alpha={alpha}, k={k}: "
              f"DW-PMAD Accuracy={acc_dw_pmad}, PCA Accuracy={acc_pca}, Better Method={better_method}")
        results_list.append([dim, target_ratio, b, alpha, k, acc_dw_pmad, acc_pca, better_method])

# Parameter settings
b_values = [1,10,30,50, 75, 100]
k_values = [3, 6, 10, 15]
alpha_values = [1, 5, 10, 25, 50]
dimensions = [28]  # Example dimension; adjust as needed
target_dims = [0.1, 0.2, 0.4]

# Load data
training_vectors = load_vectors('training_vectors_300_HIggs.npy')
testing_vectors = load_vectors('testing_vectors_1000_Higgs.npy')

# Generate all unique parameter combinations
param_combinations = list(itertools.product(dimensions, target_dims, b_values, alpha_values))

# Use multiprocessing to parallelize
if __name__ == '__main__':
    manager = mp.Manager()
    results_list = manager.list()

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(process_parameters, [(params, results_list) for params in param_combinations])

    # Convert results to DataFrame and export
    results_df = pd.DataFrame(list(results_list),
                              columns=['Dimension', 'Target Ratio', 'b', 'alpha', 'k',
                                       'DW-PMAD Accuracy', 'PCA Accuracy', 'Better Method'])
    results_df.to_csv('parameter_sweep_results_Higgsb%closest.csv', index=False)
    print(results_df)
    print("Results exported to 'parameter_sweep_results_Higgsb%closest.csv'")
