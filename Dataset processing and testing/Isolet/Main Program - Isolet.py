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

    # Debugging output

    return -np.mean(np.partition(abs_diffs, top_b_count)[:top_b_count])  # Partial sort

# Orthogonality constraint
def orthogonality_constraint(w, prev_ws, alpha):
    return sum((np.dot(w, prev_w) ** 2) for prev_w in prev_ws)


# Optimized DW-PMAD
def dw_pmad(X, b, alpha, target_dim):
    X_centered = X - np.mean(X, axis=0)
    prev_ws, optimal_ws = [], []

    for axis in range(target_dim):
        def constrained_dw_pmad(w):
            return dw_pmad_b(w, X_centered, b) + alpha * orthogonality_constraint(w, prev_ws, alpha)

        result = minimize(constrained_dw_pmad, np.random.randn(X.shape[1]), method='L-BFGS-B')
        optimal_w = result.x / np.linalg.norm(result.x)

        prev_ws.append(optimal_w)
        optimal_ws.append(optimal_w)

        # print(f"DW-PMAD axis {axis + 1}/{target_dim} computed successfully.")

    return X_centered @ np.column_stack(optimal_ws), np.column_stack(optimal_ws)


# Project test data using stored DW-PMAD axes
def project_dw_pmad(X, projection_axes):
    return (X - np.mean(X, axis=0)) @ projection_axes


# Accuracy calculation
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
b_values = [35,50,70,85, 100]
k_values = [3,6, 10,15]
alpha_values = [1,5,10,15,25,50]
dimensions = [50] # 617?
target_dims = [0.05, 0.1,0.2, 0.4, 0.6]

# Load data
training_vectors = load_vectors('training_vectors_300_Isolet.npy')
testing_vectors = load_vectors('testing_vectors_1000_Isolet.npy')

# Generate all unique parameter combinations
param_combinations = list(itertools.product(dimensions, target_dims, b_values, alpha_values))

# Use multiprocessing to parallelize
if __name__ == '__main__':
    manager = mp.Manager()
    results_list = manager.list()

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(process_parameters, [(params, results_list) for params in param_combinations])

    # Convert results to DataFrame and export
    results_df = pd.DataFrame(list(results_list), columns=['Dimension', 'Target Ratio', 'b', 'alpha', 'k', 'DW-PMAD Accuracy', 'PCA Accuracy', 'Better Method'])
    results_df.to_csv('parameter_sweep_results_Isolet.csv', index=False)
    print(results_df)
    print("Results exported to 'parameter_sweep_results_CIFAR-10_Multiple_methodsPast.csv'")
