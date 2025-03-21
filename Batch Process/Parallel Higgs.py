import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool  # Use threads for inner parallelism
import time
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist  # kept for comparison if needed
from tqdm import tqdm
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
import umap  # Standard import; ensure this is from umap-learn

# ------------------------------
# Helper Functions for DW-PMAD Parallelism
# ------------------------------

def compute_diff_range(args):
    # Compute pairwise absolute differences for indices in [start, end)
    vector, start, end = args
    N = len(vector)
    diffs = []
    for i in range(start, end):
        for j in range(i+1, N):
            diffs.append(abs(vector[i] - vector[j]))
    return np.array(diffs)

def linear_index_to_pair(indices, N):
    """
    Convert linear indices of the upper triangular matrix (excluding the diagonal)
    into pair indices (i, j) with i < j, using vectorized operations.
    """
    # For each row i, the number of pairs is (N - i - 1)
    counts = np.cumsum(np.arange(N - 1, 0, -1))
    # Find for each linear index the corresponding row index i
    i = np.searchsorted(counts, indices, side='right')
    # Determine the starting count for the found row
    prev_counts = np.zeros_like(i)
    prev_counts[i > 0] = counts[i[i > 0] - 1]
    # Compute column index j
    j = indices - prev_counts + i + 1
    return i, j

def parallel_pdist(vector, sample_fraction=1):
    """
    Compute the absolute differences of all pairs (i, j) for i < j in 'vector'.

    To improve computational efficiency, this function employs:
      1. Vectorized operations: Uses NumPy's advanced indexing to compute differences.
      3. Approximation algorithms: Instead of computing all pairs, it computes a random
         subset of pairs. 'sample_fraction' controls the trade-off between speed and accuracy.
         The approximation error (for example, in quantile estimation) is roughly O(1/sqrt(sample_size)).
      5. Efficient data structures: Avoids Python loops by converting a set of linear indices
         (representing positions in the upper triangular difference matrix) into (i, j) pairs.

    Parameters:
      vector (np.ndarray): 1D array of numbers.
      sample_fraction (float): Fraction of all pairwise differences to compute (in (0, 1]).
          Use 1.0 for exact computation; lower values yield an approximation.

    Returns:
      np.ndarray: Array of absolute differences (either exact or approximated).
    """
    N = len(vector)
    total_pairs = N * (N - 1) // 2

    if sample_fraction >= 1.0 or total_pairs < 1e4:
        # For small arrays or full computation, compute all differences vectorized.
        i, j = np.triu_indices(N, k=1)
        return np.abs(vector[i] - vector[j])
    else:
        # Determine sample size based on desired fraction.
        sample_size = int(total_pairs * sample_fraction)
        # Randomly sample 'sample_size' linear indices from all possible pair positions.
        linear_indices = np.random.choice(total_pairs, size=sample_size, replace=False)
        # Convert these linear indices into pair indices (i, j) using a vectorized mapping.
        i, j = linear_index_to_pair(linear_indices, N)
        return np.abs(vector[i] - vector[j])

# ------------------------------
# Load Pre-trained Vectors
# ------------------------------

def load_vectors(file_path):
    return np.load(file_path)

# ------------------------------
# DW-PMAD Calculation (with timing and parallel inner loop)
# ------------------------------

def dw_pmad_b(w, X, b):
    # Normalize w
    w = w / np.linalg.norm(w)

    # --- Parallel Projection ---
    projections = X @ w  # Vectorized projection (1D array)

    # --- Parallel Pairwise Distance Calculation ---
    abs_diffs = parallel_pdist(projections)

    # --- Parallel Sorting (using np.partition which is efficient) ---
    num_pairs = len(abs_diffs)
    top_b_count = min(num_pairs - 1, max(1, int((b / 100) * num_pairs)))
    partitioned = np.partition(abs_diffs, top_b_count)[:top_b_count]

    # Return the negative mean (to be minimized)
    return -np.mean(partitioned)

def orthogonality_constraint(w, prev_ws, alpha):
    return sum((np.dot(w, prev_w) ** 2) for prev_w in prev_ws)

def dw_pmad(X, b, alpha, target_dim):
    total_start = time.perf_counter()
    prev_ws, optimal_ws = [], []
    for axis in range(target_dim):
        def constrained_dw_pmad(w):
            return dw_pmad_b(w, X, b) + alpha * orthogonality_constraint(w, prev_ws, alpha)
        # Optimize for one basis vector
        result = minimize(constrained_dw_pmad, np.random.randn(X.shape[1]), method='L-BFGS-B')
        optimal_w = result.x / np.linalg.norm(result.x)
        prev_ws.append(optimal_w)
        optimal_ws.append(optimal_w)
    # Project the data using the computed axes
    total_time = time.perf_counter() - total_start
    print(f"DW-PMAD timing: {total_time:.4f}s")
    return X @ np.column_stack(optimal_ws), np.column_stack(optimal_ws)

def project_dw_pmad(X, projection_axes):
    return X @ projection_axes

# ------------------------------
# Accuracy Calculation (Using ThreadPool for inner parallelism)
# ------------------------------

def calculate_accuracy_parallel(original_data, reduced_data, new_original_data, new_reduced_data, k):
    total_start = time.perf_counter()
    # Fit nearest neighbors models on the training (original and reduced) data.
    nbrs_original = NearestNeighbors(n_neighbors=k).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=k).fit(reduced_data)

    # Compute nearest neighbor indices for all test points at once.
    all_inds_orig = nbrs_original.kneighbors(new_original_data, return_distance=False)
    all_inds_reduced = nbrs_reduced.kneighbors(new_reduced_data, return_distance=False)

    # Define a helper function for computing the intersection count for a given index.
    def intersection_count(i):
        return len(set(all_inds_orig[i]) & set(all_inds_reduced[i]))

    # Use a ThreadPool (threads) to compute intersection counts in parallel.
    with ThreadPool(processes=mp.cpu_count() - 1) as pool:
        results = pool.map(intersection_count, range(len(new_original_data)))

    total_matches = sum(results)
    total_time = time.perf_counter() - total_start
    print(f"Parallel accuracy computation for k={k} took {total_time:.4f}s")
    return total_matches / (len(new_original_data) * k)

# ------------------------------
# Main Processing Function (for one parameter combination)
# ------------------------------

def process_parameters(params, test_results_list, use_dw_pmad):
    # Unpack parameters. For DW-PMAD, params are (dim, target_ratio, b, alpha)
    if use_dw_pmad:
        dim, target_ratio, b, alpha = params
    else:
        dim, target_ratio = params
        b, alpha = "N/A", "N/A"

    target_dim = max(1, int(dim * target_ratio))
    print(f"\nProcessing: Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}, Target Dim={target_dim}")

    # Set seed for reproducibility
    np.random.seed(1)

    # Select dimensions from the training data.
    total_dims = training_vectors.shape[1]
    selected_dims = np.random.choice(total_dims, size=dim, replace=False)
    X_train = training_vectors[:, selected_dims]
    X_test = testing_vectors[:, selected_dims]

    # Standardize the data.
    standardization_start = time.perf_counter()
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1  # Avoid division by zero
    X_train_standardized = (X_train - train_mean) / train_std
    X_test_standardized = (X_test - train_mean) / train_std
    standardization_time = time.perf_counter() - standardization_start
    print(f"Data standardization complete in {standardization_time:.4f}s")

    # --- DW-PMAD (if requested) ---
    if use_dw_pmad:
        print(f"Starting DW-PMAD for Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}")
        dw_start = time.perf_counter()
        X_dw_pmad, dw_pmad_axes = dw_pmad(X_train_standardized, b, alpha, target_dim)
        new_dw_pmad = project_dw_pmad(X_test_standardized, dw_pmad_axes)
        dw_time = time.perf_counter() - dw_start
        print(f"DW-PMAD complete in {dw_time:.4f}s")
    else:
        X_dw_pmad = None
        new_dw_pmad = None
        dw_time = np.nan

    # --- PCA ---
    pca_start = time.perf_counter()
    pca = PCA(n_components=target_dim).fit(X_train_standardized)
    X_pca = pca.transform(X_train_standardized)
    new_pca = pca.transform(X_test_standardized)
    pca_time = time.perf_counter() - pca_start
    print(f"PCA complete in {pca_time:.4f}s")

    # --- UMAP ---
    umap_start = time.perf_counter()
    umap_model = umap.UMAP(n_components=target_dim, random_state=1)
    X_umap = umap_model.fit_transform(X_train_standardized)
    new_umap = umap_model.transform(X_test_standardized)
    umap_time = time.perf_counter() - umap_start
    print(f"UMAP complete in {umap_time:.4f}s")

    # --- Isomap ---
    isomap_start = time.perf_counter()
    isomap = Isomap(n_components=target_dim)
    X_isomap = isomap.fit_transform(X_train_standardized)
    new_isomap = isomap.transform(X_test_standardized)
    isomap_time = time.perf_counter() - isomap_start
    print(f"Isomap complete in {isomap_time:.4f}s")

    # --- Kernel PCA ---
    kpca_start = time.perf_counter()
    kernel_pca = KernelPCA(n_components=target_dim, kernel='rbf', random_state=1)
    X_kernel_pca = kernel_pca.fit_transform(X_train_standardized)
    new_kernel_pca = kernel_pca.transform(X_test_standardized)
    kpca_time = time.perf_counter() - kpca_start
    print(f"Kernel PCA complete in {kpca_time:.4f}s")

    # --- MDS with Regression-based Out-of-sample Extension ---
    mds_start = time.perf_counter()
    mds = MDS(n_components=target_dim, dissimilarity='euclidean', random_state=1)
    X_mds = mds.fit_transform(X_train_standardized)
    regressor = LinearRegression().fit(X_train_standardized, X_mds)
    new_mds = regressor.predict(X_test_standardized)
    mds_time = time.perf_counter() - mds_start
    print(f"MDS complete in {mds_time:.4f}s")

    # --- Accuracy Calculations for Testing (only) ---
    accuracy_start = time.perf_counter()
    for k in k_values:
        methods = []
        if use_dw_pmad:
            acc_dw_pmad_test = calculate_accuracy_parallel(X_train_standardized, X_dw_pmad, X_test_standardized, new_dw_pmad, k)
            methods.append(('dw_pmad', acc_dw_pmad_test))
        else:
            acc_dw_pmad_test = np.nan

        acc_pca_test = calculate_accuracy_parallel(X_train_standardized, X_pca, X_test_standardized, new_pca, k)
        acc_umap_test = calculate_accuracy_parallel(X_train_standardized, X_umap, X_test_standardized, new_umap, k)
        acc_isomap_test = calculate_accuracy_parallel(X_train_standardized, X_isomap, X_test_standardized, new_isomap, k)
        acc_kernel_pca_test = calculate_accuracy_parallel(X_train_standardized, X_kernel_pca, X_test_standardized, new_kernel_pca, k)
        acc_mds_test = calculate_accuracy_parallel(X_train_standardized, X_mds, X_test_standardized, new_mds, k)

        methods.append(('pca', acc_pca_test))
        methods.append(('umap', acc_umap_test))
        methods.append(('isomap', acc_isomap_test))
        methods.append(('kernel_pca', acc_kernel_pca_test))
        methods.append(('mds', acc_mds_test))

        better_method_test = max(methods, key=lambda x: x[1])[0]
        print(f"Test Results for dim={dim}, target_ratio={target_ratio}, b={b}, alpha={alpha}, k={k}: "
              f"DW-PMAD Acc={acc_dw_pmad_test}, PCA Acc={acc_pca_test}, UMAP Acc={acc_umap_test}, "
              f"Isomap Acc={acc_isomap_test}, KernelPCA Acc={acc_kernel_pca_test}, MDS Acc={acc_mds_test}, "
              f"Best: {better_method_test}")
        # Record testing results along with timing info for this k.
        test_results_list.append([dim, target_ratio, b, alpha, k,
                                  acc_dw_pmad_test, acc_pca_test, acc_umap_test,
                                  acc_isomap_test, acc_kernel_pca_test, acc_mds_test,
                                  better_method_test,
                                  dw_time, pca_time, umap_time, isomap_time, kpca_time, mds_time])
    accuracy_time = time.perf_counter() - accuracy_start
    print(f"Accuracy calculations complete in {accuracy_time:.4f}s")

# ------------------------------
# Parameter Settings and Data Loading
# ------------------------------

# Parameter settings
b_values = [60,70,80,90,100]      # Used if DW-PMAD is enabled
k_values = [1, 3, 6, 10, 15]
alpha_values = [1,6,12,18,25,35,50,10000]       # Used if DW-PMAD is enabled
dimensions = [28]         # Example dimension; adjust as needed
target_dims = [0.05, 0.1, 0.2, 0.4, 0.6]

# Load data
training_vectors = load_vectors('training_vectors_600_Higgs.npy')
testing_vectors = load_vectors('testing_vectors_600_Higgs.npy')

if __name__ == '__main__':
    total_start = time.perf_counter()
    use_dw_input = "y".strip().lower()
    use_dw_pmad = True if use_dw_input.startswith('y') else False

    # Create parameter combinations.
    if use_dw_pmad:
        param_combinations = list(itertools.product(dimensions, target_dims, b_values, alpha_values))
    else:
        param_combinations = list(itertools.product(dimensions, target_dims))

    # Use a Manager list to collect results from all parameter combinations.
    manager = mp.Manager()
    test_results_list = manager.list()

    overall_start = time.perf_counter()
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        pool.starmap(process_parameters, [(params, test_results_list, use_dw_pmad) for params in param_combinations])
    overall_time = time.perf_counter() - overall_start
    print(f"Overall processing complete in {overall_time:.4f}s")

    # Define column names (with additional timing columns)
    columns = ['Dimension', 'Target Ratio', 'b', 'alpha', 'k',
               'DW-PMAD Accuracy', 'PCA Accuracy', 'UMAP Accuracy',
               'Isomap Accuracy', 'KernelPCA Accuracy', 'MDS Accuracy',
               'Better Method',
               'DW-PMAD Time (s)', 'PCA Time (s)', 'UMAP Time (s)', 'Isomap Time (s)',
               'KernelPCA Time (s)', 'MDS Time (s)']

    # Save test results.
    test_results_df = pd.DataFrame(list(test_results_list), columns=columns)
    test_results_df.to_csv('parameter_sweep_results_Higgs_Multiple_methods.csv', index=False)
    print(test_results_df)
    print("Test results exported to 'parameter_sweep_Higgs_Fasttext_Multiple_methods.csv'")
    total_time = time.perf_counter() - total_start
    print(f"Total time is {total_time:.4f}s")
