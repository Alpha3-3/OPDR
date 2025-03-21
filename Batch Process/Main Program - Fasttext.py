import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from tqdm import tqdm
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
import umap  # Standard import; ensure this is from umap-learn

# Load pre-trained vectors
def load_vectors(file_path):
    return np.load(file_path)

# DW-PMAD calculation
def dw_pmad_b(w, X, b):
    w = w / np.linalg.norm(w)  # Normalize direction vector
    projections = X @ w
    abs_diffs = pdist(projections.reshape(-1, 1))  # Efficient pairwise differences
    num_pairs = len(abs_diffs)
    top_b_count = min(num_pairs - 1, max(1, int((b / 100) * num_pairs)))
    return -np.mean(np.partition(abs_diffs, top_b_count)[:top_b_count])  # Partial sort

# Orthogonality constraint for DW-PMAD
def orthogonality_constraint(w, prev_ws, alpha):
    return sum((np.dot(w, prev_w) ** 2) for prev_w in prev_ws)

# DW-PMAD routine (assumes X is already standardized)
def dw_pmad(X, b, alpha, target_dim):
    prev_ws, optimal_ws = [], []
    for axis in range(target_dim):
        def constrained_dw_pmad(w):
            return dw_pmad_b(w, X, b) + alpha * orthogonality_constraint(w, prev_ws, alpha)
        result = minimize(constrained_dw_pmad, np.random.randn(X.shape[1]), method='L-BFGS-B')
        optimal_w = result.x / np.linalg.norm(result.x)
        prev_ws.append(optimal_w)
        optimal_ws.append(optimal_w)
    # Project the data using the computed axes
    return X @ np.column_stack(optimal_ws), np.column_stack(optimal_ws)

# Project test data using stored DW-PMAD axes.
def project_dw_pmad(X, projection_axes):
    return X @ projection_axes

# Accuracy calculation based on nearest neighbors (for testing vectors).
def calculate_accuracy(original_data, reduced_data, new_original_data, new_reduced_data, k):
    nbrs_original = NearestNeighbors(n_neighbors=k).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=k).fit(reduced_data)
    total_matches = sum(
        len(set(nbrs_original.kneighbors(new_original_data[i].reshape(1, -1), return_distance=False)[0]) &
            set(nbrs_reduced.kneighbors(new_reduced_data[i].reshape(1, -1), return_distance=False)[0]))
        for i in range(len(new_original_data))
    )
    return total_matches / (len(new_original_data) * k)

# --- NEW FUNCTION ---
# Calculate training accuracy by testing on every training vector.
# Because each vector is in the reference set, we use k+1 neighbors and then remove the self-match.
def calculate_training_accuracy(original_data, reduced_data, k):
    nbrs_original = NearestNeighbors(n_neighbors=k+1).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=k+1).fit(reduced_data)
    total_matches = 0
    for i in range(len(original_data)):
        # Get k+1 neighbors (which include the point itself)
        indices_orig = nbrs_original.kneighbors(original_data[i].reshape(1, -1), return_distance=False)[0]
        indices_reduced = nbrs_reduced.kneighbors(reduced_data[i].reshape(1, -1), return_distance=False)[0]
        # Remove self (if present) and then take the first k neighbors.
        indices_orig = [j for j in indices_orig if j != i][:k]
        indices_reduced = [j for j in indices_reduced if j != i][:k]
        total_matches += len(set(indices_orig) & set(indices_reduced))
    return total_matches / (len(original_data) * k)

# Worker function for multiprocessing that processes one set of parameters.
def process_parameters(params, test_results_list, training_results_list, use_dw_pmad):
    # Unpack parameters based on whether DW-PMAD is used.
    if use_dw_pmad:
        dim, target_ratio, b, alpha = params
    else:
        dim, target_ratio = params
        b, alpha = "N/A", "N/A"

    target_dim = max(1, int(dim * target_ratio))
    print(f"Processing: Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}, Target Dim={target_dim}")

    # Set seed for reproducibility
    np.random.seed(1)

    # Select dimensions from the training data.
    total_dims = training_vectors.shape[1]
    selected_dims = np.random.choice(total_dims, size=dim, replace=False)
    X_train = training_vectors[:, selected_dims]
    X_test = testing_vectors[:, selected_dims]

    # Standardize the data.
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1  # Avoid division by zero
    X_train_standardized = (X_train - train_mean) / train_std
    X_test_standardized = (X_test - train_mean) / train_std

    # --- DW-PMAD (if requested) ---
    if use_dw_pmad:
        print(f"Starting DW-PMAD for Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}")
        X_dw_pmad, dw_pmad_axes = dw_pmad(X_train_standardized, b, alpha, target_dim)
        new_dw_pmad = project_dw_pmad(X_test_standardized, dw_pmad_axes)
        print(f"DW-PMAD complete for Dimension={dim}, Target Ratio={target_ratio}, b={b}, alpha={alpha}")
    else:
        X_dw_pmad = None
        new_dw_pmad = None

    # --- PCA ---
    print(f"Starting PCA for Dimension={dim}, Target Ratio={target_ratio}")
    pca = PCA(n_components=target_dim).fit(X_train_standardized)
    X_pca = pca.transform(X_train_standardized)
    new_pca = pca.transform(X_test_standardized)
    print(f"PCA complete for Dimension={dim}, Target Ratio={target_ratio}")

    # --- Additional Dimension Reduction Methods ---

    # UMAP
    umap_model = umap.UMAP(n_components=target_dim, random_state=1)
    X_umap = umap_model.fit_transform(X_train_standardized)
    new_umap = umap_model.transform(X_test_standardized)
    print(f"UMAP complete for Dimension={dim}, Target Ratio={target_ratio}")

    # Isomap
    isomap = Isomap(n_components=target_dim)
    X_isomap = isomap.fit_transform(X_train_standardized)
    new_isomap = isomap.transform(X_test_standardized)
    print(f"Isomap complete for Dimension={dim}, Target Ratio={target_ratio}")

    # Kernel PCA
    kernel_pca = KernelPCA(n_components=target_dim, kernel='rbf', random_state=1)
    X_kernel_pca = kernel_pca.fit_transform(X_train_standardized)
    new_kernel_pca = kernel_pca.transform(X_test_standardized)
    print(f"Kernel PCA complete for Dimension={dim}, Target Ratio={target_ratio}")

    # MDS with regression-based out-of-sample extension
    mds = MDS(n_components=target_dim, dissimilarity='euclidean', random_state=1)
    X_mds = mds.fit_transform(X_train_standardized)
    regressor = LinearRegression().fit(X_train_standardized, X_mds)
    new_mds = regressor.predict(X_test_standardized)
    print(f"MDS complete for Dimension={dim}, Target Ratio={target_ratio}")

    # --- Accuracy Calculations for Testing (B) ---
    for k in k_values:
        methods = []
        if use_dw_pmad:
            acc_dw_pmad_test = calculate_accuracy(X_train_standardized, X_dw_pmad, X_test_standardized, new_dw_pmad, k)
            methods.append(('dw_pmad', acc_dw_pmad_test))
        else:
            acc_dw_pmad_test = np.nan

        acc_pca_test = calculate_accuracy(X_train_standardized, X_pca, X_test_standardized, new_pca, k)
        acc_umap_test = calculate_accuracy(X_train_standardized, X_umap, X_test_standardized, new_umap, k)
        acc_isomap_test = calculate_accuracy(X_train_standardized, X_isomap, X_test_standardized, new_isomap, k)
        acc_kernel_pca_test = calculate_accuracy(X_train_standardized, X_kernel_pca, X_test_standardized, new_kernel_pca, k)
        acc_mds_test = calculate_accuracy(X_train_standardized, X_mds, X_test_standardized, new_mds, k)

        methods.append(('pca', acc_pca_test))
        methods.append(('umap', acc_umap_test))
        methods.append(('isomap', acc_isomap_test))
        methods.append(('kernel_pca', acc_kernel_pca_test))
        methods.append(('mds', acc_mds_test))

        better_method_test = max(methods, key=lambda x: x[1])[0]
        print(f"Test Results for dim={dim}, target_ratio={target_ratio}, b={b}, alpha={alpha}, k={k}: "
              f"DW-PMAD Accuracy={acc_dw_pmad_test}, PCA Accuracy={acc_pca_test}, UMAP Accuracy={acc_umap_test}, "
              f"Isomap Accuracy={acc_isomap_test}, KernelPCA Accuracy={acc_kernel_pca_test}, MDS Accuracy={acc_mds_test}, "
              f"Better Method={better_method_test}")
        test_results_list.append([dim, target_ratio, b, alpha, k,
                                  acc_dw_pmad_test, acc_pca_test, acc_umap_test, acc_isomap_test, acc_kernel_pca_test, acc_mds_test,
                                  better_method_test])

    # --- NEW: Accuracy Calculations for Training (A) ---
    for k in k_values:
        methods_train = []
        if use_dw_pmad:
            acc_dw_pmad_train = calculate_training_accuracy(X_train_standardized, X_dw_pmad, k)
            methods_train.append(('dw_pmad', acc_dw_pmad_train))
        else:
            acc_dw_pmad_train = np.nan

        acc_pca_train = calculate_training_accuracy(X_train_standardized, X_pca, k)
        acc_umap_train = calculate_training_accuracy(X_train_standardized, X_umap, k)
        acc_isomap_train = calculate_training_accuracy(X_train_standardized, X_isomap, k)
        acc_kernel_pca_train = calculate_training_accuracy(X_train_standardized, X_kernel_pca, k)
        acc_mds_train = calculate_training_accuracy(X_train_standardized, X_mds, k)

        methods_train.append(('pca', acc_pca_train))
        methods_train.append(('umap', acc_umap_train))
        methods_train.append(('isomap', acc_isomap_train))
        methods_train.append(('kernel_pca', acc_kernel_pca_train))
        methods_train.append(('mds', acc_mds_train))

        better_method_train = max(methods_train, key=lambda x: x[1])[0]
        print(f"Training Results for dim={dim}, target_ratio={target_ratio}, b={b}, alpha={alpha}, k={k}: "
              f"DW-PMAD Accuracy={acc_dw_pmad_train}, PCA Accuracy={acc_pca_train}, UMAP Accuracy={acc_umap_train}, "
              f"Isomap Accuracy={acc_isomap_train}, KernelPCA Accuracy={acc_kernel_pca_train}, MDS Accuracy={acc_mds_train}, "
              f"Better Method={better_method_train}")
        training_results_list.append([dim, target_ratio, b, alpha, k,
                                      acc_dw_pmad_train, acc_pca_train, acc_umap_train, acc_isomap_train, acc_kernel_pca_train, acc_mds_train,
                                      better_method_train])

# Parameter settings
b_values = [50]  # Only used if DW-PMAD is enabled
k_values = [1,3, 6, 10, 15]
alpha_values = [1,2,3,4]  # Only used if DW-PMAD is enabled
dimensions = [50]   # Example dimension; adjust as needed
target_dims = [0.05, 0.1, 0.2, 0.4, 0.6]

# Load data
training_vectors = load_vectors('training_vectors_600_Fasttext.npy')
testing_vectors = load_vectors('testing_vectors_1000_Fasttext.npy')

if __name__ == '__main__':
    # Ask the user if DW-PMAD should be used.
    # use_dw_input = input("Do you want to use DW-PMAD? (y/n): ").strip().lower()
    # Hard coded
    use_dw_input = "y".strip().lower()
    use_dw_pmad = True if use_dw_input.startswith('y') else False

    # Create parameter combinations depending on DW-PMAD usage.
    if use_dw_pmad:
        param_combinations = list(itertools.product(dimensions, target_dims, b_values, alpha_values))
    else:
        param_combinations = list(itertools.product(dimensions, target_dims))

    manager = mp.Manager()
    test_results_list = manager.list()
    training_results_list = manager.list()

    # Pass the use_dw_pmad flag and both result lists to each worker.
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(process_parameters, [(params, test_results_list, training_results_list, use_dw_pmad) for params in param_combinations])

    # Define column names (same for both accuracy types)
    columns = ['Dimension', 'Target Ratio', 'b', 'alpha', 'k',
               'DW-PMAD Accuracy', 'PCA Accuracy', 'UMAP Accuracy',
               'Isomap Accuracy', 'KernelPCA Accuracy', 'MDS Accuracy',
               'Better Method']

    # Save test (B) results.
    test_results_df = pd.DataFrame(list(test_results_list), columns=columns)
    test_results_df.to_csv('parameter_sweep_results_Fasttext_Multiple_methods.csv', index=False)
    print(test_results_df)
    print("Test results exported to 'parameter_sweep_results_Fasttext_Multiple_methods.csv'")

    # Save training (A) results.
    training_results_df = pd.DataFrame(list(training_results_list), columns=columns)
    training_results_df.to_csv('training_accuracy_results_Fasttext_Multiple_methods.csv', index=False)
    print(training_results_df)
    print("Training results exported to 'training_accuracy_results_Fasttext_Multiple_methods.csv'")
