import numpy as np
import pandas as pd
import time
# import itertools # No longer needed
from sklearn.neighbors import NearestNeighbors
# from sklearn.decomposition import FastICA, NMF # No longer needed
# from sklearn.manifold import TSNE, LocallyLinearEmbedding # No longer needed
from sklearn.random_projection import GaussianRandomProjection
# from sklearn.cluster import FeatureAgglomeration # No longer needed
# from sklearn.linear_model import LinearRegression # No longer needed for tSNE out-of-sample
import os

# For autoencoder and VAE - will be removed as per request
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras import backend as K

# -------------------------
# k-NN accuracy calculation (same as in the base code)
# -------------------------
def calculate_accuracy(original_data, reduced_data, new_original_data, new_reduced_data, k):
    total_start = time.perf_counter()
    # Ensure data is C-contiguous for NearestNeighbors
    original_data = np.ascontiguousarray(original_data)
    reduced_data = np.ascontiguousarray(reduced_data)
    new_original_data = np.ascontiguousarray(new_original_data)
    new_reduced_data = np.ascontiguousarray(new_reduced_data)

    nbrs_original = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(original_data)
    # For LSH, if data is binary, metric='hamming' might be more appropriate,
    # but default 'minkowski' with p=2 (Euclidean) on binary data is proportional to Hamming.
    nbrs_reduced = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(reduced_data)

    total_matches = 0
    for i in range(len(new_original_data)):
        # Ensure reshape for single sample
        query_orig_reshaped = new_original_data[i].reshape(1, -1)
        query_red_reshaped = new_reduced_data[i].reshape(1, -1)

        inds_orig = nbrs_original.kneighbors(query_orig_reshaped,
                                             return_distance=False)[0]
        inds_reduced = nbrs_reduced.kneighbors(query_red_reshaped,
                                               return_distance=False)[0]
        total_matches += len(set(inds_orig) & set(inds_reduced))
    total_time = time.perf_counter() - total_start
    # For diagnostic printing:
    print(f"Accuracy calc time for k={k}: {total_time:.4f}s")
    return total_matches / (len(new_original_data) * k)

# -------------------------
# Baseline method implementations
# -------------------------
def run_random_projection(X_train, X_test, target_dim):
    from sklearn.random_projection import GaussianRandomProjection # Already imported globally
    t0 = time.perf_counter()
    rp = GaussianRandomProjection(n_components=target_dim, random_state=1)
    X_train_rp = rp.fit_transform(X_train)
    X_test_rp = rp.transform(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_rp, X_test_rp, t_elapsed

def run_lsh(X_train, X_test, target_dim):
    """
    Implements LSH by using random projections and binarizing the output.
    target_dim here represents the number of hash bits (hyperplanes).
    """
    t0 = time.perf_counter()
    # Initialize a Gaussian Random Projection
    # The number of components will be our number of hash bits
    lsh_projector = GaussianRandomProjection(n_components=target_dim, random_state=42)

    # Fit on training data and transform both train and test
    X_train_projected = lsh_projector.fit_transform(X_train)
    X_test_projected = lsh_projector.transform(X_test)

    # Binarize the projections to get hash codes
    # (values > 0 become 1, others become 0)
    X_train_lsh = (X_train_projected > 0).astype(int)
    X_test_lsh = (X_test_projected > 0).astype(int)

    t_elapsed = time.perf_counter() - t0
    return X_train_lsh, X_test_lsh, t_elapsed

# Dictionary to map method names to functions
# Remove all other methods and add LSH
methods_mapping = {
    'RandomProjection': run_random_projection, # Kept as a basic baseline
    'LSH': run_lsh,
}

# -------------------------
# Main pipeline over multiple datasets
# -------------------------
# Define your dataset file paths.
# (Adjust the file names as needed for your environment.)
datasets = {
    'Arcene': ('training_vectors_600_Arcene.npy', 'testing_vectors_300_Arcene.npy'),
    'Fasttext': ('training_vectors_600_Fasttext.npy', 'testing_vectors_300_Fasttext.npy'),
    'Isolet': ('training_vectors_600_Isolet.npy', 'testing_vectors_300_Isolet.npy'),
    'PBMC3k': ('training_vectors_600_PBMC3k.npy', 'testing_vectors_300_PBMC3k.npy')
    # Add other datasets here if needed, e.g.
    # 'Gisette': ('gisette_train.data.npy', 'gisette_test.data.npy'),
    # 'Fasttext': ('fasttext_train.npy', 'fasttext_test.npy')
}

# Base parameter for feature selection for most datasets is 200,
# but for Fasttext use 298.
default_dim = 200

target_ratios = [0.05, 0.1, 0.2, 0.4, 0.6] # These will determine num_hash_bits for LSH
# Target dimensions will be computed based on the dim used for that dataset.
k_values = [1, 3, 6, 10, 15]

# Loop over each dataset
for dname, (train_path, test_path) in datasets.items():
    print(f"\nProcessing dataset: {dname}")
    # Adjust sampling dim for Fasttext
    if dname == 'Fasttext':
        cur_dim = 300 # Or the actual dimensionality if known
    else:
        cur_dim = default_dim

    # Load the dataset (assumes .npy files)
    # This is a placeholder for actual loading logic if files don't exist
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Files for dataset {dname} not found. Generating dummy data for demonstration.")
        # Generate dummy data if files are not found
        num_train_samples = 600
        num_test_samples = 300
        original_features = 10000 if dname == 'Arcene' else 5000 # Example feature sizes

        training_vectors = np.random.rand(num_train_samples, original_features)
        testing_vectors = np.random.rand(num_test_samples, original_features)
        # Save them so they can be "found" on a re-run if desired
        # np.save(train_path, training_vectors)
        # np.save(test_path, testing_vectors)
    else:
        training_vectors = np.load(train_path)
        testing_vectors = np.load(test_path)

    print(f"Original training data shape: {training_vectors.shape}")
    print(f"Original testing data shape: {testing_vectors.shape}")


    # Randomly select "cur_dim" features (if the data dimensionality is higher)
    total_dims = training_vectors.shape[1]
    if total_dims > cur_dim:
        print(f"Selecting {cur_dim} features from original {total_dims} features.")
        np.random.seed(1) # for reproducibility
        selected_dims = np.random.choice(total_dims, size=cur_dim, replace=False)
        X_train = training_vectors[:, selected_dims]
        X_test = testing_vectors[:, selected_dims]
    else:
        print(f"Using all {total_dims} features as it's not greater than cur_dim {cur_dim}.")
        X_train = training_vectors
        X_test = testing_vectors

    print(f"Shape of X_train after feature selection: {X_train.shape}")
    print(f"Shape of X_test after feature selection: {X_test.shape}")

    # Standardize the data (using training mean and std)
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1  # avoid division by zero
    X_train_std = (X_train - train_mean) / train_std
    X_test_std = (X_test - train_mean) / train_std

    # Compute target dimensions for this dataset based on cur_dim
    # For LSH, these target_dims will be the number of hash bits
    target_dims_for_methods = [max(1, int(X_train_std.shape[1] * r)) for r in target_ratios]
    # Ensure target_dim does not exceed original dim for RandomProjection
    # For LSH, target_dim (num_bits) can theoretically be anything, but usually less than original_dim.

    print(f"Target dimensions (or num_hash_bits for LSH) to be tested: {target_dims_for_methods}")

    results_rows = []
    # Loop over each method and target dimension combination
    for method_name, method_func in methods_mapping.items():
        for target_dim_val in target_dims_for_methods:
            # Ensure target_dim for RandomProjection is not > number of features
            if method_name == 'RandomProjection' and target_dim_val > X_train_std.shape[1]:
                print(f"Skipping {method_name} for target_dim {target_dim_val} as it exceeds original dim {X_train_std.shape[1]}")
                continue

            current_target_dim = target_dim_val

            print(f"Method: {method_name}, TargetDim/NumHashBits: {current_target_dim}")
            X_train_red, X_test_red, method_time = method_func(X_train_std, X_test_std, current_target_dim)

            print(f"Shape of reduced training data: {X_train_red.shape}")
            print(f"Shape of reduced testing data: {X_test_red.shape}")

            # For each k value, compute k-NN accuracy.
            for k_val in k_values:
                # Ensure k is not greater than the number of training samples
                if k_val > X_train_red.shape[0]:
                    print(f"Skipping k={k_val} as it's larger than the number of samples in reduced training data ({X_train_red.shape[0]})")
                    continue

                acc = calculate_accuracy(X_train_std, X_train_red, X_test_std, X_test_red, k_val)
                row = {
                    'Dataset': dname,
                    'Method': method_name,
                    'OriginalDim': X_train_std.shape[1], # Actual dimension after initial selection
                    'TargetDim_NumHashBits': current_target_dim,
                    'k': k_val,
                    'Accuracy': acc,
                    'MethodTime(s)': method_time
                }
                results_rows.append(row)

    # Save the results for this dataset to a CSV file.
    if results_rows: # Check if there are any results to save
        df_results = pd.DataFrame(results_rows)
        csv_filename = f"lsh_baseline_results_{dname}.csv"
        df_results.to_csv(csv_filename, index=False)
        print(f"Results for dataset {dname} saved to {csv_filename}")
    else:
        print(f"No results generated for dataset {dname}.")

print("\nAll dataset processing completed.")