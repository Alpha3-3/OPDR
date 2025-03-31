import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
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

# ---------------------------------------------------------
# GLOBAL POOL & HELPER FUNCTIONS FOR PARALLEL DISTANCE
# ---------------------------------------------------------
global_pool = None

def init_global_pool(num_workers=None):
    """
    Initialize a single global pool so we can reuse it for parallel distance
    computations and avoid overhead or nested multiprocessing.
    """
    global global_pool
    if global_pool is None:
        global_pool = mp.Pool(processes=(num_workers or mp.cpu_count()))

def linear_index_to_pair(indices, N):
    """
    Convert linear indices of the upper triangular matrix (excluding the diagonal)
    into pair indices (i, j) with i < j, using vectorized operations.
    """
    counts = np.cumsum(np.arange(N - 1, 0, -1))
    i = np.searchsorted(counts, indices, side='right')
    prev_counts = np.zeros_like(i)
    idx_gt_zero = (i > 0)
    prev_counts[idx_gt_zero] = counts[i[idx_gt_zero] - 1]
    j = indices - prev_counts + i + 1
    return i, j

def compute_diff_range_chunk(chunk_indices, vector):
    """
    Compute abs differences for pairs in chunk_indices.
    """
    i, j = linear_index_to_pair(np.array(chunk_indices), len(vector))
    return np.abs(vector[i] - vector[j])

def parallel_pdist(vector, sample_fraction=1.0):
    """
    Compute pairwise absolute differences with limited precision (float32).
    The vector is converted to float32 before any calculations.
    """
    # Limit precision to float32 for pairwise distance calculations
    vector = vector.astype(np.float32, copy=False)
    N = len(vector)
    total_pairs = N * (N - 1) // 2

    if sample_fraction >= 1.0 or total_pairs < 1e4:
        i, j = np.triu_indices(N, k=1)
        return np.abs(vector[i] - vector[j])
    else:
        sample_size = int(total_pairs * sample_fraction)
        linear_indices = np.random.choice(total_pairs, size=sample_size, replace=False)
        num_cores = mp.cpu_count()
        chunk_size = max(1, sample_size // num_cores)
        chunks = [linear_indices[i:i + chunk_size] for i in range(0, sample_size, chunk_size)]
        results = global_pool.starmap(
            compute_diff_range_chunk,
            [(chunk, vector) for chunk in chunks]
        )
        return np.concatenate(results)

# ---------------------------------------------------------
# LOAD VECTORS
# ---------------------------------------------------------

def load_vectors(file_path):
    return np.load(file_path)

# ---------------------------------------------------------
# DW-PMAD CALCULATIONS (UNCHANGED EXCEPT FOR parallel_pdist)
# ---------------------------------------------------------

def dw_pmad_b(w, X, b):
    # Normalize w
    w = w / np.linalg.norm(w)

    # Project data
    projections = X @ w  # projections will initially be of X's dtype (often float64)
    # Pairwise distance calculations now use float32 precision inside parallel_pdist
    abs_diffs = parallel_pdist(projections)

    # Partition to get top (b%) differences
    num_pairs = len(abs_diffs)
    top_b_count = min(num_pairs - 1, max(1, int((b / 100) * num_pairs)))
    partitioned = np.partition(abs_diffs, top_b_count)[:top_b_count]

    # Return negative mean for minimization
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
    total_time = time.perf_counter() - total_start
    print(f"DW-PMAD timing: {total_time:.4f}s")
    return X @ np.column_stack(optimal_ws), np.column_stack(optimal_ws)

def project_dw_pmad(X, projection_axes):
    return X @ projection_axes

# ---------------------------------------------------------
# ACCURACY CALCULATION (k-NN with n_jobs=-1 for parallelism)
# ---------------------------------------------------------

def calculate_accuracy(original_data, reduced_data, new_original_data, new_reduced_data, k):
    total_start = time.perf_counter()
    nbrs_original = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(reduced_data)
    total_matches = 0
    for i in range(len(new_original_data)):
        inds_orig = nbrs_original.kneighbors(new_original_data[i].reshape(1, -1),
                                             return_distance=False)[0]
        inds_reduced = nbrs_reduced.kneighbors(new_reduced_data[i].reshape(1, -1),
                                               return_distance=False)[0]
        total_matches += len(set(inds_orig) & set(inds_reduced))
    total_time = time.perf_counter() - total_start
    print(f"Time for calculating accuracy for k= {k} is {total_time:.4f}s")
    return total_matches / (len(new_original_data) * k)

# ---------------------------------------------------------
# PROCESSING FUNCTION (RUNS IN *ONE* PROCESS AT A TIME)
# ---------------------------------------------------------

def process_parameters(params, test_results_list, use_dw_pmad):
    """
    Runs the pipeline for one parameter set.
    """
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

    # --- Accuracy Calculations ---
    accuracy_start = time.perf_counter()
    for k in k_values:
        methods = []
        if use_dw_pmad:
            acc_dw_pmad_test = calculate_accuracy(X_train_standardized, X_dw_pmad,
                                                  X_test_standardized, new_dw_pmad, k)
            methods.append(('dw_pmad', acc_dw_pmad_test))
        else:
            acc_dw_pmad_test = np.nan

        acc_pca_test = calculate_accuracy(X_train_standardized, X_pca,
                                          X_test_standardized, new_pca, k)
        acc_umap_test = calculate_accuracy(X_train_standardized, X_umap,
                                           X_test_standardized, new_umap, k)
        acc_isomap_test = calculate_accuracy(X_train_standardized, X_isomap,
                                             X_test_standardized, new_isomap, k)
        acc_kernel_pca_test = calculate_accuracy(X_train_standardized, X_kernel_pca,
                                                 X_test_standardized, new_kernel_pca, k)
        acc_mds_test = calculate_accuracy(X_train_standardized, X_mds,
                                          X_test_standardized, new_mds, k)

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
        test_results_list.append([
            dim, target_ratio, b, alpha, k,
            acc_dw_pmad_test, acc_pca_test, acc_umap_test,
            acc_isomap_test, acc_kernel_pca_test, acc_mds_test,
            better_method_test,
            dw_time, pca_time, umap_time, isomap_time, kpca_time, mds_time
        ])
    accuracy_time = time.perf_counter() - accuracy_start
    print(f"Accuracy calculations complete in {accuracy_time:.4f}s")


# ---------------------------------------------------------
# MAIN EXECUTION (PARAM COMBINATIONS IN A SEQUENTIAL LOOP)
# ---------------------------------------------------------

# Parameter settings
b_values = [60]
k_values = [1, 3, 6, 10, 15]
alpha_values = [1, 6, 12, 18, 25, 35, 50, 10000]
dimensions = [28]  # Example dimension
target_dims = [0.05, 0.1, 0.2, 0.4, 0.6]

# Load data
training_vectors = load_vectors('training_vectors_600_Higgs.npy')
testing_vectors = load_vectors('testing_vectors_600_Higgs.npy')

if __name__ == '__main__':
    total_start = time.perf_counter()
    print(b_values, k_values, alpha_values, dimensions, target_dims)

    # Decide whether to use DW-PMAD
    use_dw_input = "y".strip().lower()
    use_dw_pmad = True if use_dw_input.startswith('y') else False

    # Create parameter combinations
    if use_dw_pmad:
        param_combinations = list(itertools.product(dimensions, target_dims, b_values, alpha_values))
    else:
        param_combinations = list(itertools.product(dimensions, target_dims))

    # Initialize a global pool for internal parallel tasks
    init_global_pool()

    test_results_list = []
    for params in param_combinations:
        process_parameters(params, test_results_list, use_dw_pmad)

    # Close out the pool
    global_pool.close()
    global_pool.join()

    # Define column names
    columns = [
        'Dimension', 'Target Ratio', 'b', 'alpha', 'k',
        'DW-PMAD Accuracy', 'PCA Accuracy', 'UMAP Accuracy',
        'Isomap Accuracy', 'KernelPCA Accuracy', 'MDS Accuracy',
        'Better Method',
        'DW-PMAD Time (s)', 'PCA Time (s)', 'UMAP Time (s)', 'Isomap Time (s)',
        'KernelPCA Time (s)', 'MDS Time (s)'
    ]

    # Save test results
    test_results_df = pd.DataFrame(test_results_list, columns=columns)
    test_results_df.to_csv('parameter_sweep_results_Higgs_Multiple_method60.csv', index=False)
    print(test_results_df)
    print("Test results exported to 'parameter_sweep_results_Higgs_Multiple_methods60.csv'")

    total_time = time.perf_counter() - total_start
    print(f"Total time is {total_time:.4f}s")
