import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
import time
import os # For performance report
import psutil # For memory usage
from scipy.optimize import minimize
from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF
from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import pdist # Kept for comparison if needed, but parallel_pdist is used
from tqdm import tqdm
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
from sklearn.linear_model import LinearRegression
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
import umap # Standard import; ensure this is from umap-learn
import faiss # For ANN methods

# For autoencoder and VAE (from baseline methods)
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# ---------------------------------------------------------
# GLOBAL POOL & HELPER FUNCTIONS FOR PARALLEL DISTANCE
# ---------------------------------------------------------
global_pool = None

def init_global_pool(num_workers=None):
    global global_pool
    if global_pool is None:
        cpu_cores = num_workers or mp.cpu_count()
        global_pool = mp.Pool(processes=cpu_cores)
        try:
            faiss.omp_set_num_threads(cpu_cores)
            print(f"Global pool initialized with {cpu_cores} workers. Faiss OMP threads set to {cpu_cores}.")
        except AttributeError: # Handles older Faiss versions or different compile options
            print(f"Global pool initialized with {cpu_cores} workers. Faiss OMP settings not available/changed.")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    try:
        return process.memory_info().uss # Unique Set Size
    except AttributeError:
        # print("Warning: USS memory metric not available, falling back to RSS.")
        return process.memory_info().rss # Resident Set Size

def linear_index_to_pair(indices, N):
    counts = np.cumsum(np.arange(N - 1, 0, -1))
    i = np.searchsorted(counts, indices, side='right')
    prev_counts = np.zeros_like(i)
    idx_gt_zero = (i > 0)
    prev_counts[idx_gt_zero] = counts[i[idx_gt_zero] - 1]
    j = indices - prev_counts + i + 1
    return i, j

def compute_diff_range_chunk(chunk_indices, vector):
    i, j = linear_index_to_pair(np.array(chunk_indices), len(vector))
    return np.abs(vector[i] - vector[j])

def parallel_pdist(vector, sample_fraction=1.0):
    vector = vector.astype(np.float32, copy=False)
    N = len(vector)
    total_pairs = N * (N - 1) // 2

    if total_pairs == 0:
        return np.array([], dtype=np.float32)

    if sample_fraction >= 1.0 or total_pairs < 1e4:
        i, j = np.triu_indices(N, k=1)
        return np.abs(vector[i] - vector[j])
    else:
        sample_size = int(total_pairs * sample_fraction)
        sample_size = max(1, min(sample_size, total_pairs))

        linear_indices = np.random.choice(total_pairs, size=sample_size, replace=False)

        if global_pool is None:
            # print("Warning: global_pool not initialized before parallel_pdist call. Initializing now.")
            init_global_pool() # Fallback initialization

        num_pool_processes = global_pool._processes if global_pool and hasattr(global_pool, '_processes') else mp.cpu_count()
        chunk_size = max(1, sample_size // num_pool_processes)
        chunks = [linear_indices[i:i + chunk_size] for i in range(0, sample_size, chunk_size)]

        valid_chunks = [chunk for chunk in chunks if len(chunk) > 0]
        if not valid_chunks:
            return np.array([], dtype=np.float32)

        results = global_pool.starmap(
            compute_diff_range_chunk,
            [(chunk, vector) for chunk in valid_chunks]
        )
        if not results:
            return np.array([], dtype=np.float32)
        return np.concatenate(results)

# ---------------------------------------------------------
# LOAD VECTORS
# ---------------------------------------------------------

def load_vectors(file_path):
    return np.load(file_path).astype(np.float32)

# ---------------------------------------------------------
# DW-PMAD CALCULATIONS
# ---------------------------------------------------------
def dw_pmad_b(w, X, b_percentage):
    w = w / (np.linalg.norm(w) + 1e-9)
    projections = X @ w
    abs_diffs = parallel_pdist(projections)

    if len(abs_diffs) == 0:
        return 0.0

    num_pairs = len(abs_diffs)
    top_b_count = min(max(1, int((b_percentage / 100) * num_pairs)), num_pairs)

    if top_b_count == num_pairs:
        partitioned = abs_diffs
    else:
        partitioned_idx = max(0, top_b_count -1) # Correct index for partitioning
        partitioned = np.partition(abs_diffs, partitioned_idx)[:top_b_count]


    return -np.mean(partitioned) if len(partitioned) > 0 else 0.0


def orthogonality_constraint(w, prev_ws):
    penalty = 0.0
    for prev_w_val in prev_ws:
        penalty += (np.dot(w, prev_w_val) ** 2)
    return penalty


def dw_pmad(X, b_percentage_val, alpha_penalty_val, target_dim_val):
    total_start_time = time.perf_counter()
    prev_ws_list, optimal_ws_list = [], []
    n_features = X.shape[1]

    optimizer_options = {'maxiter': 25000, 'maxfun': 25000, 'ftol': 1e-8, 'gtol': 1e-6}

    for axis_num in range(target_dim_val):
        def objective_function(w_opt):
            w_norm = w_opt / (np.linalg.norm(w_opt) + 1e-9) # Normalize w
            main_objective = dw_pmad_b(w_norm, X, b_percentage_val)
            ortho_penalty_val_calc = orthogonality_constraint(w_norm, prev_ws_list)
            return main_objective + alpha_penalty_val * ortho_penalty_val_calc

        initial_w_guess = np.random.randn(n_features).astype(np.float32)
        initial_w_guess /= (np.linalg.norm(initial_w_guess) + 1e-9) # Normalize initial guess

        result = minimize(objective_function, initial_w_guess, method='L-BFGS-B', options=optimizer_options)

        if result.success:
            optimal_w_found = result.x / (np.linalg.norm(result.x) + 1e-9) # Normalize final w
        else:
            print(f"Warning: DW-PMAD optimization for axis {axis_num+1} did not converge optimally (Message: {result.message}). Using resulting vector.")
            optimal_w_found = result.x / (np.linalg.norm(result.x) + 1e-9) # Still normalize

        prev_ws_list.append(optimal_w_found.astype(np.float32))
        optimal_ws_list.append(optimal_w_found.astype(np.float32))

    projection_axes_found = np.column_stack(optimal_ws_list)
    projected_X_result = X @ projection_axes_found
    total_time_taken = time.perf_counter() - total_start_time
    # print(f"DW-PMAD timing: {total_time_taken:.4f}s") # Reduced verbosity, summary print in process_parameters
    return projected_X_result, projection_axes_found


def project_dw_pmad(X_to_project, projection_axes_to_use):
    return X_to_project @ projection_axes_to_use

# ---------------------------------------------------------
# Baseline method implementations (copied from your provided code)
# ---------------------------------------------------------
def run_random_projection(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    rp = GaussianRandomProjection(n_components=target_dim, random_state=1)
    X_train_rp = rp.fit_transform(X_train)
    X_test_rp = rp.transform(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_rp, X_test_rp, t_elapsed

def run_fastica(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    ica = FastICA(n_components=target_dim, random_state=1, max_iter=200, tol=0.01) # Default max_iter=200
    X_train_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)
    t_elapsed = time.perf_counter() - t0
    return X_train_ica, X_test_ica, t_elapsed

def run_tsne(X_train, X_test, target_dim):
    # t-SNE is usually not applied to a separate test set in the same way as other DR methods.
    # The common practice is to fit t-SNE on the combined dataset or just the training set for visualization.
    # Out-of-sample extension is non-trivial. The provided linear regression is a simple approximation.
    t0 = time.perf_counter()
    # Ensure target_dim for t-SNE is typically small (2 or 3)
    # The method='exact' can be very slow for large N. Consider 'barnes_hut' for larger datasets.
    # For very high dimensional data, PCA pre-processing to ~50 dims is often recommended for t-SNE.
    tsne_model = TSNE(n_components=target_dim, method='exact', random_state=1, perplexity=30.0, n_iter=1000)

    # If X_train is too small for perplexity, TSNE will error.
    # Perplexity should be less than n_samples - 1.
    if X_train.shape[0] -1 < tsne_model.perplexity:
        print(f"t-SNE Warning: n_samples ({X_train.shape[0]}) is too small for perplexity ({tsne_model.perplexity}). Adjusting perplexity.")
        tsne_model.perplexity = max(5, X_train.shape[0] - 2) # Adjust to a smaller valid value

    if X_train.shape[0] <= tsne_model.n_components: # n_samples must be > n_components
        print(f"t-SNE Skipping: n_samples ({X_train.shape[0]}) <= n_components ({tsne_model.n_components}).")
        nan_shape_train = (X_train.shape[0], target_dim)
        nan_shape_test = (X_test.shape[0], target_dim)
        return np.full(nan_shape_train, np.nan), np.full(nan_shape_test, np.nan), 0.0

    X_train_tsne = tsne_model.fit_transform(X_train)

    # Out-of-sample extension via linear regression (simple approximation)
    if X_train.shape[0] > 0 and X_train_tsne.shape[0] > 0 :
        reg = LinearRegression().fit(X_train, X_train_tsne)
        X_test_tsne = reg.predict(X_test)
    else:
        X_test_tsne = np.full((X_test.shape[0], target_dim), np.nan)

    t_elapsed = time.perf_counter() - t0
    return X_train_tsne, X_test_tsne, t_elapsed

def run_nmf(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    # NMF requires non-negative data; shift data accordingly.
    # Important: This shift should ideally be learned from X_train and applied to X_test,
    # or use a global min if data characteristics are similar.
    # The original code used global_min = min(X_train.min(), X_test.min())
    # For a strict train/test split, it's better to use X_train.min()

    train_min = X_train.min()
    X_train_nmf_shifted = X_train - train_min
    X_test_nmf_shifted = X_test - train_min

    # If after shifting, some values are still slightly negative due to floating point, clip them.
    X_train_nmf_shifted[X_train_nmf_shifted < 0] = 0
    X_test_nmf_shifted[X_test_nmf_shifted < 0] = 0

    nmf_model = NMF(n_components=target_dim, init='random', random_state=1, max_iter=200, tol=1e-4) # Default max_iter=200

    try:
        X_train_nmf_trans = nmf_model.fit_transform(X_train_nmf_shifted)
        X_test_nmf_trans = nmf_model.transform(X_test_nmf_shifted)
    except ValueError as e:
        print(f"NMF Error: {e}. This can happen if data is not strictly non-negative even after shift. Returning NaNs.")
        nan_shape_train = (X_train.shape[0], target_dim)
        nan_shape_test = (X_test.shape[0], target_dim)
        return np.full(nan_shape_train, np.nan), np.full(nan_shape_test, np.nan), time.perf_counter() - t0

    t_elapsed = time.perf_counter() - t0
    return X_train_nmf_trans, X_test_nmf_trans, t_elapsed

def run_lle(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    # n_neighbors for LLE should be > target_dim
    n_neighbors_lle = max(target_dim + 1, min(10, X_train.shape[0] -1))
    if X_train.shape[0] <= n_neighbors_lle or X_train.shape[0] <= target_dim :
        print(f"LLE Skipping: n_samples ({X_train.shape[0]}) is too small for n_neighbors ({n_neighbors_lle}) or target_dim ({target_dim}).")
        nan_shape_train = (X_train.shape[0], target_dim)
        nan_shape_test = (X_test.shape[0], target_dim)
        return np.full(nan_shape_train, np.nan), np.full(nan_shape_test, np.nan), 0.0

    lle_model = LocallyLinearEmbedding(n_components=target_dim, n_neighbors=n_neighbors_lle, random_state=1, method='standard')
    X_train_lle = lle_model.fit_transform(X_train)
    X_test_lle = lle_model.transform(X_test) # LLE transform can be unstable for new data
    t_elapsed = time.perf_counter() - t0
    return X_train_lle, X_test_lle, t_elapsed

def run_feature_agglomeration(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    if target_dim > X_train.shape[1]: # n_clusters cannot be > n_features
        print(f"FeatureAgglomeration Skipping: target_dim ({target_dim}) > n_features ({X_train.shape[1]})")
        nan_shape_train = (X_train.shape[0], X_train.shape[1]) # Output will be same as input if skipped this way
        nan_shape_test = (X_test.shape[0], X_test.shape[1])
        # Or return NaNs of target_dim, but FA usually means reducing features.
        # Let's return NaNs of original shape if it truly skips, or try to proceed.
        # For now, if target_dim is too high, it will fail. Let sklearn handle it or cap target_dim.
        # Capping target_dim to X_train.shape[1] if it's higher.
        actual_target_dim = min(target_dim, X_train.shape[1])
        if actual_target_dim <=0 : actual_target_dim = 1 # ensure at least 1 cluster
    else:
        actual_target_dim = target_dim

    if actual_target_dim == 0 and X_train.shape[1] > 0: actual_target_dim = 1 # Ensure at least 1 cluster if features exist

    if X_train.shape[1] == 0: # No features to agglomerate
        X_train_fa = np.zeros((X_train.shape[0], 0))
        X_test_fa = np.zeros((X_test.shape[0], 0))
    elif actual_target_dim > 0 :
        fa_model = FeatureAgglomeration(n_clusters=actual_target_dim)
        X_train_fa = fa_model.fit_transform(X_train)
        X_test_fa = fa_model.transform(X_test)
    else: # Should not happen with guard above
        X_train_fa = np.zeros((X_train.shape[0], 0))
        X_test_fa = np.zeros((X_test.shape[0], 0))

    t_elapsed = time.perf_counter() - t0
    return X_train_fa, X_test_fa, t_elapsed

def run_autoencoder(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    input_dim = X_train.shape[1]
    if input_dim == 0: # No input features
        return np.zeros((X_train.shape[0], target_dim)), np.zeros((X_test.shape[0], target_dim)), 0.0

    tf.keras.utils.set_random_seed(1) # For reproducibility

    inputs = Input(shape=(input_dim,))
    encoded = Dense(target_dim, activation='relu', name='encoder_layer')(inputs) # Named layer
    decoded = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(inputs, decoded, name='autoencoder_model')
    encoder = Model(inputs, encoded, name='encoder_model') # Access encoder part

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    # Train the autoencoder (epochs can be increased for better performance)
    # Consider adding a validation split if X_train is large enough, or using X_test for validation in a non-benchmark setting.
    history = autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0, shuffle=True)

    X_train_ae = encoder.predict(X_train, batch_size=256)
    X_test_ae = encoder.predict(X_test, batch_size=256)
    t_elapsed = time.perf_counter() - t0
    K.clear_session() # Clear TF session to free memory
    return X_train_ae, X_test_ae, t_elapsed

def run_vae(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    input_dim = X_train.shape[1]
    if input_dim == 0:
        return np.zeros((X_train.shape[0], target_dim)), np.zeros((X_test.shape[0], target_dim)), 0.0

    tf.keras.utils.set_random_seed(1)

    latent_dim = target_dim
    intermediate_dim = max(latent_dim * 2, 64) # Heuristic for intermediate layer size

    inputs = Input(shape=(input_dim,))
    h = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean_layer')(h) # Named layer
    z_log_var = Dense(latent_dim, name='z_log_var_layer')(h) # Named layer

    # Sampling function
    def sampling(args):
        z_mean_s, z_log_var_s = args
        batch = K.shape(z_mean_s)[0]
        dim = K.int_shape(z_mean_s)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
        return z_mean_s + K.exp(0.5 * z_log_var_s) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name='z_sampling_layer')([z_mean, z_log_var])

    # Decoder
    decoder_h_layer = Dense(intermediate_dim, activation='relu', name='decoder_h_layer')
    decoder_out_layer = Dense(input_dim, activation='sigmoid', name='decoder_out_layer') # Sigmoid if input is normalized [0,1]
    # or linear if input is standardized.
    # Given standardization, linear might be better.
    # Let's assume original code used linear for AE.

    h_decoded = decoder_h_layer(z)
    outputs = decoder_out_layer(h_decoded)

    vae = Model(inputs, outputs, name='vae_model')

    # Loss: Reconstruction + KL divergence
    reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    reconstruction_loss *= input_dim # Scale by input_dim (common practice)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    # Train VAE
    vae.fit(X_train, None, epochs=20, batch_size=32, verbose=0, shuffle=True)

    # Use the mean as the encoded representation
    encoder = Model(inputs, z_mean, name='vae_encoder_model') # Model to get z_mean
    X_train_vae = encoder.predict(X_train, batch_size=256)
    X_test_vae = encoder.predict(X_test, batch_size=256)

    t_elapsed = time.perf_counter() - t0
    K.clear_session() # Clear TF session
    return X_train_vae, X_test_vae, t_elapsed

# List of new baseline DR method names (used for conditional calls)
NEW_BASELINE_DR_METHODS = [
    'RandomProjection', 'FastICA', 'tSNE', 'NMF', 'LLE',
    'FeatureAgglomeration', 'Autoencoder', 'VAE'
]

# ---------------------------------------------------------
# FAISS HELPER (and other accuracy helpers)
# ---------------------------------------------------------
def get_valid_pq_m(d, max_m_val=8, ensure_d_multiple=True, min_subvector_dim=1):
    if d == 0: return 0
    if d < min_subvector_dim : return 0

    upper_bound_m = min(d // min_subvector_dim if min_subvector_dim > 0 else d, max_m_val, d)
    if upper_bound_m == 0 and d > 0 :
        upper_bound_m = 1

    for m_candidate in range(upper_bound_m, 0, -1):
        if ensure_d_multiple:
            if d % m_candidate == 0:
                return m_candidate
        else:
            return m_candidate
    return 0

def get_dynamic_nbits(n_train, m_pq, default_nbits=8, min_nbits=4):
    if m_pq == 0: return default_nbits
    nbits = default_nbits
    while n_train < 4 * (2**nbits) and nbits > min_nbits:
        nbits -= 1
    return max(min_nbits, nbits)


# ---------------------------------------------------------
# ACCURACY CALCULATION
# ---------------------------------------------------------

def get_exact_neighbors(data_to_index, query_data, k_neighbors):
    if data_to_index.shape[0] == 0 or query_data.shape[0] == 0:
        return np.array([[] for _ in range(query_data.shape[0])], dtype=int)

    actual_k = min(k_neighbors, data_to_index.shape[0])
    if actual_k == 0 :
        return np.array([[] for _ in range(query_data.shape[0])], dtype=int)


    nbrs_exact = NearestNeighbors(n_neighbors=actual_k, algorithm='auto', n_jobs=-1).fit(data_to_index)
    exact_indices_found = np.empty((len(query_data), actual_k), dtype=int)
    for i in range(len(query_data)):
        exact_indices_found[i] = nbrs_exact.kneighbors(query_data[i].reshape(1, -1), return_distance=False)[0]
    return exact_indices_found


def calculate_accuracy_exact_knn(exact_indices_orig_ground_truth, reduced_data_train_to_index, reduced_data_test_to_query, k_val_for_query):
    total_start = time.perf_counter()
    if reduced_data_train_to_index.shape[0] < 1 or reduced_data_test_to_query.shape[0] < 1:
        return np.nan, 0.0

    actual_k_for_reduced_query = min(k_val_for_query, reduced_data_train_to_index.shape[0])
    if actual_k_for_reduced_query == 0: return np.nan, 0.0


    nbrs_reduced = NearestNeighbors(n_neighbors=actual_k_for_reduced_query, algorithm='auto', n_jobs=-1).fit(reduced_data_train_to_index)
    total_matches_found = 0

    for i in range(len(reduced_data_test_to_query)):
        if exact_indices_orig_ground_truth.shape[1] == 0 : continue

        inds_reduced_found = nbrs_reduced.kneighbors(reduced_data_test_to_query[i].reshape(1, -1), return_distance=False)[0]
        total_matches_found += len(set(exact_indices_orig_ground_truth[i]) & set(inds_reduced_found))

    accuracy_calculated = total_matches_found / (len(reduced_data_test_to_query) * exact_indices_orig_ground_truth.shape[1]) \
        if len(reduced_data_test_to_query) > 0 and exact_indices_orig_ground_truth.shape[1] > 0 else 0.0

    total_time_taken = time.perf_counter() - total_start
    return accuracy_calculated, total_time_taken

def _generic_faiss_accuracy_calc(faiss_index_built, exact_indices_orig_ground_truth, reduced_data_test_faiss_ready, k_val_for_query):
    total_matches_faiss = 0
    if reduced_data_test_faiss_ready.shape[0] > 0 and \
            exact_indices_orig_ground_truth.shape[1] > 0 and \
            faiss_index_built is not None and faiss_index_built.ntotal > 0:

        actual_k_for_faiss_search = min(k_val_for_query, faiss_index_built.ntotal)
        if actual_k_for_faiss_search == 0: return 0.0

        _, inds_reduced_ann_found = faiss_index_built.search(reduced_data_test_faiss_ready, actual_k_for_faiss_search)

        for i in range(len(reduced_data_test_faiss_ready)):
            total_matches_faiss += len(set(exact_indices_orig_ground_truth[i]) & set(inds_reduced_ann_found[i]))

        return total_matches_faiss / (len(reduced_data_test_faiss_ready) * exact_indices_orig_ground_truth.shape[1])
    return 0.0

def calculate_accuracy_hnswflat_faiss(exact_indices_orig, reduced_data_train_faiss, reduced_data_test_faiss, k_query):
    total_start = time.perf_counter()
    dim_reduced = reduced_data_train_faiss.shape[1]
    if dim_reduced == 0 or reduced_data_train_faiss.shape[0] == 0: return np.nan, 0.0

    index = faiss.IndexHNSWFlat(dim_reduced, 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 50
    try:
        index.add(reduced_data_train_faiss)
    except RuntimeError as e:
        print(f"HNSWFlat Add Error: {e}. Skipping HNSWFlat.")
        return np.nan, time.perf_counter() - total_start

    accuracy = _generic_faiss_accuracy_calc(index, exact_indices_orig, reduced_data_test_faiss, k_query)
    total_time = time.perf_counter() - total_start
    return accuracy, total_time

def calculate_accuracy_ivfpq_faiss(exact_indices_orig, reduced_data_train_faiss, reduced_data_test_faiss, k_query):
    total_start = time.perf_counter()
    dim_reduced = reduced_data_train_faiss.shape[1]
    n_train = reduced_data_train_faiss.shape[0]
    if dim_reduced == 0 or n_train == 0: return np.nan, 0.0

    nlist = min(100, max(1, n_train // 39 if n_train // 39 > 0 else 1))

    m_pq = get_valid_pq_m(dim_reduced, max_m_val=8, min_subvector_dim=2)
    if m_pq == 0 :
        print(f"IVFPQ Skipping: Could not find valid m_pq for d={dim_reduced} with min_subvector_dim=2.")
        return np.nan, 0.0

    nbits_pq = get_dynamic_nbits(n_train, m_pq, default_nbits=8, min_nbits=4)

    quantizer = faiss.IndexFlatL2(dim_reduced)
    index = None
    try:
        index = faiss.IndexIVFPQ(quantizer, dim_reduced, nlist, m_pq, nbits_pq)
        if n_train < 39 * nlist :
            print(f"Faiss IVFPQ Warning: n_train ({n_train}) < Faiss heuristic min training points ({39*nlist} for nlist={nlist}). Quality may be affected.")

        if n_train < (2**nbits_pq) * 4 :
            print(f"Faiss IVFPQ Warning: n_train ({n_train}) may be low for PQ training (m_pq={m_pq}, nbits={nbits_pq}, {2**nbits_pq} centroids/subq).")


        index.train(reduced_data_train_faiss)
        index.add(reduced_data_train_faiss)
        index.nprobe = min(nlist, 10)
    except RuntimeError as e:
        print(f"IVFPQ Error: {e}. Params: d={dim_reduced}, nlist={nlist}, m_pq={m_pq}, nbits={nbits_pq}. Skipping IVFPQ.")
        return np.nan, time.perf_counter() - total_start

    accuracy = _generic_faiss_accuracy_calc(index, exact_indices_orig, reduced_data_test_faiss, k_query)
    total_time = time.perf_counter() - total_start
    return accuracy, total_time

def calculate_accuracy_hnswpq_faiss(exact_indices_orig, reduced_data_train_faiss, reduced_data_test_faiss, k_query):
    total_start = time.perf_counter()
    dim_reduced = reduced_data_train_faiss.shape[1]
    n_train = reduced_data_train_faiss.shape[0]

    if dim_reduced < 2 or n_train == 0:
        print(f"HNSWPQ Skipping: dim_reduced ({dim_reduced}) is < 2 or n_train ({n_train}) is 0.")
        return np.nan, 0.0

    M_hnsw = 32
    efConstruction_hnsw = 40
    efSearch_hnsw = 50

    m_pq_to_try = get_valid_pq_m(dim_reduced, max_m_val=8, min_subvector_dim=4)

    if m_pq_to_try == 0:
        m_pq_to_try = get_valid_pq_m(dim_reduced, max_m_val=8, min_subvector_dim=2)
        if m_pq_to_try == 0:
            print(f"HNSWPQ Skipping: Could not find valid m_pq for d={dim_reduced} with min_subvector_dim=2 (required for HNSWPQ's internal PQ).")
            return np.nan, 0.0


    index = None
    try:
        index = faiss.IndexHNSWPQ(dim_reduced, M_hnsw, m_pq_to_try)
        index.hnsw.efConstruction = efConstruction_hnsw
        index.hnsw.efSearch = efSearch_hnsw

        if n_train < m_pq_to_try * (2**8 / 8) :
            print(f"Faiss HNSWPQ Warning: n_train ({n_train}) might be too small for PQ training (m_pq={m_pq_to_try}, internal nbits=8).")

        index.train(reduced_data_train_faiss)
        index.add(reduced_data_train_faiss)
    except RuntimeError as e:
        if "d % M == 0" in str(e):
            print(f"HNSWPQ specific 'd % M' RuntimeError (d={dim_reduced}, m_pq={m_pq_to_try}): {e}. This index type appears unstable for this d/m_pq. Skipping HNSWPQ.")
        elif "sub-vector size" in str(e) or "pq.dsub" in str(e):
            print(f"HNSWPQ PQ-related RuntimeError (d={dim_reduced}, m_pq={m_pq_to_try}): {e}. Skipping HNSWPQ.")
        else:
            print(f"HNSWPQ Generic RuntimeError (d={dim_reduced}, m_pq={m_pq_to_try}): {e}. Skipping HNSWPQ.")
        return np.nan, time.perf_counter() - total_start

    accuracy = _generic_faiss_accuracy_calc(index, exact_indices_orig, reduced_data_test_faiss, k_query)
    total_time = time.perf_counter() - total_start
    return accuracy, total_time


def calculate_accuracy_ivfopq_faiss(exact_indices_orig, reduced_data_train_faiss, reduced_data_test_faiss, k_query):
    total_start = time.perf_counter()
    dim_reduced = reduced_data_train_faiss.shape[1]
    n_train = reduced_data_train_faiss.shape[0]
    if dim_reduced == 0 or n_train == 0: return np.nan, 0.0

    nlist = min(100, max(1, n_train // 39 if n_train // 39 > 0 else 1))

    opq_train_m = get_valid_pq_m(dim_reduced, max_m_val=min(dim_reduced, 32), ensure_d_multiple=True, min_subvector_dim=2)
    if opq_train_m == 0:
        print(f"IVFOPQ Skipping: Could not find valid opq_train_m for d={dim_reduced} with min_subvector_dim=2.")
        return np.nan, 0.0
    opq_nbits = get_dynamic_nbits(n_train, opq_train_m, default_nbits=8, min_nbits=6)

    final_pq_m = get_valid_pq_m(dim_reduced, max_m_val=8, ensure_d_multiple=True, min_subvector_dim=2)
    if final_pq_m == 0:
        print(f"IVFOPQ Skipping: Could not find valid final_pq_m for d={dim_reduced} with min_subvector_dim=2.")
        return np.nan, 0.0
    final_nbits = get_dynamic_nbits(n_train, final_pq_m, default_nbits=8, min_nbits=4)

    factory_str = f"OPQ{opq_train_m}x{opq_nbits},IVF{nlist},PQ{final_pq_m}x{final_nbits}"

    index = None
    try:
        index = faiss.index_factory(dim_reduced, factory_str)

        min_train_ivf = 39 * nlist
        min_train_opq_internal_pq = opq_train_m * (2**opq_nbits / 8)
        min_train_final_residual_pq = final_pq_m * (2**final_nbits / 8)

        if n_train < min_train_ivf :
            print(f"Faiss IVFOPQ Warning: n_train ({n_train}) may be small for IVF part (nlist={nlist}, needs ~{min_train_ivf}).")
        if n_train < min_train_opq_internal_pq :
            print(f"Faiss IVFOPQ Warning: n_train ({n_train}) may be small for OPQ's internal PQ training (opq_M={opq_train_m}, opq_nbits={opq_nbits}, needs ~{min_train_opq_internal_pq:.0f}).")
        if n_train < min_train_final_residual_pq :
            print(f"Faiss IVFOPQ Warning: n_train ({n_train}) may be small for final PQ training (final_pq_M={final_pq_m}, nbits={final_nbits}, needs ~{min_train_final_residual_pq:.0f}).")

        index.train(reduced_data_train_faiss)
        index.add(reduced_data_train_faiss)
        index.nprobe = min(nlist, 10)

    except RuntimeError as e:
        print(f"IVFOPQ (Factory) Error: {e}. Factory: '{factory_str}'. Skipping IVFOPQ.")
        return np.nan, time.perf_counter() - total_start
    except AttributeError as e_attr:
        print(f"IVFOPQ Attribute Error (likely during setup): {e_attr}. Factory: '{factory_str}'. Skipping IVFOPQ.")
        return np.nan, time.perf_counter() - total_start


    accuracy = _generic_faiss_accuracy_calc(index, exact_indices_orig, reduced_data_test_faiss, k_query)
    total_time = time.perf_counter() - total_start
    return accuracy, total_time


# ---------------------------------------------------------
# PROCESSING FUNCTION
# ---------------------------------------------------------
def process_parameters(params_tuple, use_dw_pmad_flag_param, training_vectors_full_param, testing_vectors_full_param, k_values_global_param, performance_data_collector_list, current_dataset_id=""):
    run_performance_data_dict = {'params': params_tuple, 'dataset_id': current_dataset_id}
    current_peak_memory_usage = get_memory_usage()

    if use_dw_pmad_flag_param:
        dim_from_params, target_ratio_from_params, b_from_params, alpha_from_params = params_tuple
    else:
        dim_from_params, target_ratio_from_params = params_tuple
        b_from_params, alpha_from_params = "N/A", "N/A"

    actual_dim_selected_from_orig = min(dim_from_params, training_vectors_full_param.shape[1])
    target_dim_for_dr_final = max(1, int(actual_dim_selected_from_orig * target_ratio_from_params))
    target_dim_for_dr_final = min(target_dim_for_dr_final, actual_dim_selected_from_orig)

    print(f"\nProcessing Dataset: {current_dataset_id}, Params={params_tuple}, Orig Dim Selected={actual_dim_selected_from_orig}, Final DR Target Dim={target_dim_for_dr_final}")
    run_performance_data_dict['orig_dim_selected'] = actual_dim_selected_from_orig
    run_performance_data_dict['target_dim_final_dr'] = target_dim_for_dr_final

    np.random.seed(1)

    selected_dims_indices_val = np.random.choice(training_vectors_full_param.shape[1], size=actual_dim_selected_from_orig, replace=False)
    X_train_orig_selected_data = training_vectors_full_param[:, selected_dims_indices_val]
    X_test_orig_selected_data = testing_vectors_full_param[:, selected_dims_indices_val]

    standardization_start_time = time.perf_counter()
    train_mean_val = np.mean(X_train_orig_selected_data, axis=0)
    train_std_val = np.std(X_train_orig_selected_data, axis=0)
    train_std_val[train_std_val == 0] = 1e-6
    X_train_standardized_data = (X_train_orig_selected_data - train_mean_val) / train_std_val
    X_test_standardized_data = (X_test_orig_selected_data - train_mean_val) / train_std_val
    run_performance_data_dict['time_standardization'] = time.perf_counter() - standardization_start_time
    run_performance_data_dict['mem_after_standardization'] = get_memory_usage()
    current_peak_memory_usage = max(current_peak_memory_usage, run_performance_data_dict['mem_after_standardization'])


    dr_methods_results_output_dict = {}
    dr_timings_output_dict = {}
    dr_memory_after_output_dict = {}

    # --- DW-PMAD ---
    if use_dw_pmad_flag_param:
        print(f"Starting DW-PMAD...")
        dw_start_time_val = time.perf_counter()
        current_target_dim_for_dw = min(target_dim_for_dr_final, X_train_standardized_data.shape[1])
        current_target_dim_for_dw = max(1, current_target_dim_for_dw) if X_train_standardized_data.shape[1] > 0 else 0

        if current_target_dim_for_dw > 0 and X_train_standardized_data.shape[1] > 0 :
            X_dw_pmad_reduced_train, dw_pmad_projection_axes = dw_pmad(X_train_standardized_data, b_from_params, alpha_from_params, current_target_dim_for_dw)
            new_dw_pmad_reduced_test = project_dw_pmad(X_test_standardized_data, dw_pmad_projection_axes)
        else:
            X_dw_pmad_reduced_train = np.zeros((X_train_standardized_data.shape[0], current_target_dim_for_dw), dtype=X_train_standardized_data.dtype)
            new_dw_pmad_reduced_test = np.zeros((X_test_standardized_data.shape[0], current_target_dim_for_dw), dtype=X_test_standardized_data.dtype)
            if current_target_dim_for_dw == 0 : print("DW-PMAD skipped as target dimension is 0.")


        dr_timings_output_dict['dw_pmad'] = time.perf_counter() - dw_start_time_val
        dr_methods_results_output_dict['dw_pmad'] = (X_dw_pmad_reduced_train, new_dw_pmad_reduced_test)
        dr_memory_after_output_dict['dw_pmad'] = get_memory_usage()
        current_peak_memory_usage = max(current_peak_memory_usage, dr_memory_after_output_dict['dw_pmad'])
        print(f"DW-PMAD complete in {dr_timings_output_dict['dw_pmad']:.4f}s. Output shape: {X_dw_pmad_reduced_train.shape}")
    else:
        dr_timings_output_dict['dw_pmad'] = np.nan

        # --- Other DR Baselines ---
    dr_method_configs = {
        # Sklearn style (class, params_dict)
        'pca': (PCA, {'random_state': 1}),
        'umap': (umap.UMAP, {'random_state': 1, 'min_dist': 0.1, 'n_jobs': 1}),
        'isomap': (Isomap, {}), # n_neighbors will be set dynamically
        'kernel_pca': (KernelPCA, {'kernel': 'rbf', 'random_state': 1, 'n_jobs': -1}),
        'mds': (MDS, {'dissimilarity': 'euclidean', 'random_state': 1, 'normalized_stress':'auto', 'n_jobs': -1}),
        # New style (function_name_str_or_actual_func, None indicates direct call with X_train, X_test, target_dim)
        'RandomProjection': (run_random_projection, None),
        'FastICA': (run_fastica, None),
        'tSNE': (run_tsne, None),
        'NMF': (run_nmf, None),
        'LLE': (run_lle, None),
        'FeatureAgglomeration': (run_feature_agglomeration, None),
        'Autoencoder': (run_autoencoder, None),
        'VAE': (run_vae, None),
    }

    for dr_method_name_key, (callable_or_class, params_or_none) in dr_method_configs.items():
        if dr_method_name_key == 'dw_pmad': continue # Already processed

        print(f"Starting {dr_method_name_key.upper()}...")
        dr_method_loop_start_time = time.perf_counter() # For fallback timing if method errors early

        # Adjust n_components based on method and data properties - THIS IS CRUCIAL
        current_n_components_for_dr = target_dim_for_dr_final # Base target dimension
        if X_train_standardized_data.shape[1] == 0: # No input features
            current_n_components_for_dr = 0
        elif dr_method_name_key == 'pca':
            current_n_components_for_dr = min(target_dim_for_dr_final, X_train_standardized_data.shape[0], X_train_standardized_data.shape[1])
        elif dr_method_name_key in ['umap', 'isomap', 'mds', 'LLE', 'tSNE']: # Manifold methods
            if X_train_standardized_data.shape[1] > 1:
                current_n_components_for_dr = min(target_dim_for_dr_final, X_train_standardized_data.shape[1] -1 if dr_method_name_key != 'tSNE' else X_train_standardized_data.shape[1]) # tSNE can output same dim
                if dr_method_name_key == 'tSNE': # tSNE usually to 2 or 3
                    current_n_components_for_dr = min(current_n_components_for_dr, 3)

            else:
                current_n_components_for_dr = min(target_dim_for_dr_final, 1)
        # For NMF, FastICA, RP, FA, Autoencoder, VAE, target_dim_for_dr_final is generally fine unless > n_features
        elif dr_method_name_key in ['NMF', 'FastICA', 'RandomProjection', 'FeatureAgglomeration']:
            current_n_components_for_dr = min(target_dim_for_dr_final, X_train_standardized_data.shape[1])


        current_n_components_for_dr = max(1, current_n_components_for_dr) if X_train_standardized_data.shape[1] > 0 and current_n_components_for_dr > 0 else 0
        if X_train_standardized_data.shape[1] == 0 : current_n_components_for_dr = 0 # if no features, target dim must be 0

        skip_dr = False
        X_reduced_train_val, X_reduced_test_val = None, None

        if X_train_standardized_data.shape[1] == 0:
            skip_dr = True
            X_reduced_train_val = np.zeros((X_train_standardized_data.shape[0], 0), dtype=X_train_standardized_data.dtype)
            X_reduced_test_val = np.zeros((X_test_standardized_data.shape[0], 0), dtype=X_test_standardized_data.dtype)
        elif current_n_components_for_dr == 0 :
            skip_dr = True
            print(f"Skipping {dr_method_name_key.upper()}: Target dimension is 0.")
            X_reduced_train_val = np.zeros((X_train_standardized_data.shape[0], 0), dtype=X_train_standardized_data.dtype)
            X_reduced_test_val = np.zeros((X_test_standardized_data.shape[0], 0), dtype=X_test_standardized_data.dtype)
        elif X_train_standardized_data.shape[0] <= current_n_components_for_dr and \
                dr_method_name_key in ['umap', 'isomap', 'mds', 'LLE', 'tSNE']: # Check n_samples for manifold
            print(f"Skipping {dr_method_name_key.upper()}: n_samples ({X_train_standardized_data.shape[0]}) <= n_components ({current_n_components_for_dr}).")
            skip_dr = True
        elif dr_method_name_key == 'mds' and X_train_standardized_data.shape[0] > 3000:
            print(f"Skipping MDS: Too many samples ({X_train_standardized_data.shape[0]} > 3000) for practical computation.")
            skip_dr = True

        # LLE specific n_neighbors check (must be > n_components)
        if not skip_dr and dr_method_name_key == 'LLE':
            n_neighbors_lle_check = max(current_n_components_for_dr + 1, 5) # LLE n_neighbors > n_components
            if X_train_standardized_data.shape[0] <= n_neighbors_lle_check:
                print(f"Skipping {dr_method_name_key.upper()}: n_samples ({X_train_standardized_data.shape[0]}) too small for n_neighbors for LLE.")
                skip_dr = True


        if skip_dr and X_reduced_train_val is None:
            target_dim_for_fallback_skip = current_n_components_for_dr if current_n_components_for_dr > 0 else 1
            X_reduced_train_val = np.full((X_train_standardized_data.shape[0], target_dim_for_fallback_skip), np.nan)
            X_reduced_test_val = np.full((X_test_standardized_data.shape[0], target_dim_for_fallback_skip), np.nan)
            dr_timings_output_dict[dr_method_name_key] = time.perf_counter() - dr_method_loop_start_time
        elif not skip_dr:
            if dr_method_name_key in NEW_BASELINE_DR_METHODS:
                dr_func_to_call = callable_or_class
                try:
                    # These functions return X_train_red, X_test_red, t_elapsed
                    X_reduced_train_val, X_reduced_test_val, specific_method_time = dr_func_to_call(
                        X_train_standardized_data,
                        X_test_standardized_data,
                        current_n_components_for_dr
                    )
                    dr_timings_output_dict[dr_method_name_key] = specific_method_time
                except Exception as e_dr_baseline:
                    print(f"{dr_method_name_key.upper()} (baseline func) error: {e_dr_baseline}. Filling with NaNs.")
                    target_dim_fallback = current_n_components_for_dr if current_n_components_for_dr > 0 else 1
                    X_reduced_train_val = np.full((X_train_standardized_data.shape[0], target_dim_fallback), np.nan)
                    X_reduced_test_val = np.full((X_test_standardized_data.shape[0], target_dim_fallback), np.nan)
                    dr_timings_output_dict[dr_method_name_key] = time.perf_counter() - dr_method_loop_start_time
            else: # Existing sklearn-style DR methods
                model_class_val = callable_or_class
                model_params_dict = params_or_none.copy() if params_or_none else {}

                method_execution_start_time = time.perf_counter()
                try:
                    model_params_dict['n_components'] = current_n_components_for_dr

                    if dr_method_name_key in ['umap', 'isomap']:
                        n_neigh = min(15, X_train_standardized_data.shape[0] - 1 if X_train_standardized_data.shape[0] > 1 else 1)
                        model_params_dict['n_neighbors'] = max(1, n_neigh)
                        if X_train_standardized_data.shape[0] <= model_params_dict['n_neighbors']: # Pre-check for UMAP/Isomap
                            print(f"Skipping {dr_method_name_key.upper()}: n_samples ({X_train_standardized_data.shape[0]}) <= n_neighbors ({model_params_dict['n_neighbors']}).")
                            raise ValueError("n_samples <= n_neighbors")


                    model_instance = model_class_val(**model_params_dict)
                    X_reduced_train_val = model_instance.fit_transform(X_train_standardized_data)

                    if X_reduced_train_val.ndim == 1 and current_n_components_for_dr == 1:
                        X_reduced_train_val = X_reduced_train_val.reshape(-1,1)

                    if dr_method_name_key == 'mds':
                        if X_train_standardized_data.shape[0] > 0 and X_reduced_train_val.shape[0] > 0 :
                            regressor_mds = LinearRegression().fit(X_train_standardized_data, X_reduced_train_val)
                            X_reduced_test_val = regressor_mds.predict(X_test_standardized_data)
                            if X_reduced_test_val.ndim == 1 and current_n_components_for_dr == 1:
                                X_reduced_test_val = X_reduced_test_val.reshape(-1,1)
                        else:
                            X_reduced_test_val = np.full((X_test_standardized_data.shape[0], X_reduced_train_val.shape[1] if X_reduced_train_val.ndim >1 and X_reduced_train_val.shape[1] > 0 else 1), np.nan)
                    else:
                        X_reduced_test_val = model_instance.transform(X_test_standardized_data)
                        if X_reduced_test_val.ndim == 1 and current_n_components_for_dr == 1:
                            X_reduced_test_val = X_reduced_test_val.reshape(-1,1)
                except Exception as e_dr_sklearn:
                    print(f"{dr_method_name_key.upper()} (sklearn) error: {e_dr_sklearn}. Filling with NaNs.")
                    target_dim_fallback = model_params_dict.get('n_components', current_n_components_for_dr if current_n_components_for_dr > 0 else 1)
                    X_reduced_train_val = np.full((X_train_standardized_data.shape[0], target_dim_fallback), np.nan)
                    X_reduced_test_val = np.full((X_test_standardized_data.shape[0], target_dim_fallback), np.nan)
                dr_timings_output_dict[dr_method_name_key] = time.perf_counter() - method_execution_start_time

        dr_methods_results_output_dict[dr_method_name_key] = (X_reduced_train_val, X_reduced_test_val)
        dr_memory_after_output_dict[dr_method_name_key] = get_memory_usage()
        current_peak_memory_usage = max(current_peak_memory_usage, dr_memory_after_output_dict[dr_method_name_key])
        output_shape_str = X_reduced_train_val.shape if X_reduced_train_val is not None else "DR_Failed/Skipped"
        print(f"{dr_method_name_key.upper()} complete in {dr_timings_output_dict.get(dr_method_name_key, np.nan):.4f}s. Output shape: {output_shape_str}")


    run_performance_data_dict['dr_timings'] = dr_timings_output_dict
    run_performance_data_dict['dr_memory_after'] = dr_memory_after_output_dict

    # --- Accuracy Calculation ---
    accuracy_results_collected_dict = {k_val: {} for k_val in k_values_global_param}
    accuracy_times_collected_dict = {k_val: {} for k_val in k_values_global_param}

    all_k_exact_indices_ground_truth_dict = {}
    if X_train_standardized_data.shape[0] > 0 and X_test_standardized_data.shape[0] > 0 :
        for k_val_gt in k_values_global_param:
            all_k_exact_indices_ground_truth_dict[k_val_gt] = get_exact_neighbors(X_train_standardized_data, X_test_standardized_data, k_val_gt)
    else:
        for k_val_gt in k_values_global_param:
            all_k_exact_indices_ground_truth_dict[k_val_gt] = np.array([[] for _ in range(X_test_standardized_data.shape[0])], dtype=int)



    ann_accuracy_functions_dict = {
        "Exact_kNN": calculate_accuracy_exact_knn,
        "HNSWFlat_Faiss": calculate_accuracy_hnswflat_faiss,
        "IVFPQ_Faiss": calculate_accuracy_ivfpq_faiss,
        "HNSWPQ_Faiss": calculate_accuracy_hnswpq_faiss,
        "IVFOPQ_Faiss": calculate_accuracy_ivfopq_faiss,
    }

    for k_val_iter_acc in k_values_global_param:
        exact_k_indices_for_current_k_val = all_k_exact_indices_ground_truth_dict[k_val_iter_acc]

        if exact_k_indices_for_current_k_val.size == 0 and X_test_standardized_data.shape[0] > 0 :
            for acc_method_name_iter_nan in ann_accuracy_functions_dict:
                accuracy_results_collected_dict[k_val_iter_acc][acc_method_name_iter_nan] = {dr_name_nan: np.nan for dr_name_nan in dr_methods_results_output_dict}
                accuracy_times_collected_dict[k_val_iter_acc][acc_method_name_iter_nan] = {dr_name_nan: 0.0 for dr_name_nan in dr_methods_results_output_dict}
            continue

        for dr_method_name_iter_acc, reduced_data_tuple in dr_methods_results_output_dict.items():
            if reduced_data_tuple is None:
                continue
            X_reduced_train_iter, X_reduced_test_iter = reduced_data_tuple

            if X_reduced_train_iter is None or X_reduced_test_iter is None or \
                    np.isnan(X_reduced_train_iter).all() or np.isnan(X_reduced_test_iter).all() or \
                    X_reduced_train_iter.shape[1] == 0:
                for acc_method_name_fill_nan in ann_accuracy_functions_dict:
                    if acc_method_name_fill_nan not in accuracy_results_collected_dict[k_val_iter_acc]: accuracy_results_collected_dict[k_val_iter_acc][acc_method_name_fill_nan] = {}
                    if acc_method_name_fill_nan not in accuracy_times_collected_dict[k_val_iter_acc]: accuracy_times_collected_dict[k_val_iter_acc][acc_method_name_fill_nan] = {}
                    accuracy_results_collected_dict[k_val_iter_acc][acc_method_name_fill_nan][dr_method_name_iter_acc] = np.nan
                    accuracy_times_collected_dict[k_val_iter_acc][acc_method_name_fill_nan][dr_method_name_iter_acc] = 0.0
                continue

            X_reduced_train_faiss_ready_iter = np.ascontiguousarray(X_reduced_train_iter, dtype=np.float32)
            X_reduced_test_faiss_ready_iter = np.ascontiguousarray(X_reduced_test_iter, dtype=np.float32)


            for acc_method_name_val, acc_func_val in ann_accuracy_functions_dict.items():
                if acc_method_name_val not in accuracy_results_collected_dict[k_val_iter_acc]: accuracy_results_collected_dict[k_val_iter_acc][acc_method_name_val] = {}
                if acc_method_name_val not in accuracy_times_collected_dict[k_val_iter_acc]: accuracy_times_collected_dict[k_val_iter_acc][acc_method_name_val] = {}

                if "Faiss" in acc_method_name_val:
                    acc_calculated_val, time_calculated_val = acc_func_val(exact_k_indices_for_current_k_val, X_reduced_train_faiss_ready_iter, X_reduced_test_faiss_ready_iter, k_val_iter_acc)
                else:
                    acc_calculated_val, time_calculated_val = acc_func_val(exact_k_indices_for_current_k_val, X_reduced_train_iter, X_reduced_test_iter, k_val_iter_acc)

                accuracy_results_collected_dict[k_val_iter_acc][acc_method_name_val][dr_method_name_iter_acc] = acc_calculated_val
                accuracy_times_collected_dict[k_val_iter_acc][acc_method_name_val][dr_method_name_iter_acc] = time_calculated_val

    run_performance_data_dict['accuracy_results'] = accuracy_results_collected_dict
    run_performance_data_dict['accuracy_times'] = accuracy_times_collected_dict
    run_performance_data_dict['mem_after_accuracy'] = get_memory_usage()
    current_peak_memory_usage = max(current_peak_memory_usage, run_performance_data_dict['mem_after_accuracy'])
    run_performance_data_dict['peak_memory_in_run'] = current_peak_memory_usage

    performance_data_collector_list.append(run_performance_data_dict)


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == '__main__':
    overall_start_time_script_exec = time.perf_counter()
    initial_memory_at_script_start = get_memory_usage()

    # --- SCALABILITY TEST CONFIGURATION for Isolet ---
    # This name is used to form part of the dataset identifier and in messages.
    CURRENT_DATASET_NAME = "Fasttext"

    training_files_to_test = [
        f"training_vectors_300_{CURRENT_DATASET_NAME}.npy",
        f"training_vectors_600_{CURRENT_DATASET_NAME}.npy",
        f"training_vectors_900_{CURRENT_DATASET_NAME}.npy",
        f"training_vectors_1200_{CURRENT_DATASET_NAME}.npy"
    ]

    # Fixed parameters for this series of runs (Isolet)
    fixed_alpha = 6
    fixed_b = 90
    fixed_target_ratio = 0.6
    fixed_k_values = [1, 3, 6, 10, 15]
    fixed_original_dimension = 200 # Number of features to select from input data

    test_file_global = f'testing_vectors_300_{CURRENT_DATASET_NAME}.npy'

    # Create a parameter string for filenames
    # Sanitize float for filename: 0.1 -> "0p1"
    ratio_str_for_filename = str(fixed_target_ratio).replace('.', 'p')
    params_string_for_filename = (
        f"origD{fixed_original_dimension}_"
        f"ratio{ratio_str_for_filename}_"
        f"alpha{fixed_alpha}_"
        f"b{fixed_b}"
    )

    # Dummy file creation (if needed)
    # Feature dimension for dummy Fasttext files (should match or exceed fixed_original_dimension)
    # If your actual Fasttext files have a different native dimension, this might be adjusted.
    dummy_feature_dim_PBMC3k = fixed_original_dimension

    for f_name in training_files_to_test:
        if not os.path.exists(f_name):
            print(f"Creating dummy file: {f_name} with {dummy_feature_dim_PBMC3k} features.")
            try:
                num_samples = int(f_name.split('_')[2]) # Extracts '300', '600', etc.
                dummy_data = np.random.rand(num_samples, dummy_feature_dim_PBMC3k).astype(np.float32)
                np.save(f_name, dummy_data)
            except Exception as e:
                print(f"Could not create dummy file {f_name}: {e}")

    if not os.path.exists(test_file_global):
        print(f"Creating dummy file: {test_file_global} with {dummy_feature_dim_PBMC3k} features.")
        try:
            # Assuming test set size is 300 for dummy creation if not specified otherwise
            dummy_data_test = np.random.rand(300, dummy_feature_dim_PBMC3k).astype(np.float32)
            np.save(test_file_global, dummy_data_test)
        except Exception as e:
            print(f"Could not create dummy file {test_file_global}: {e}")

    # Set up fixed parameters for the experiment runs (used by process_parameters)
    b_values_config = [fixed_b]
    alpha_values_config = [fixed_alpha]
    k_values_for_knn_config = fixed_k_values
    dimensions_to_select_config = [fixed_original_dimension]
    target_ratios_for_dr_config = [fixed_target_ratio]

    use_dw_pmad_main_execution_flag = True

    print(f"Global testing data: {test_file_global}")
    try:
        if not os.path.exists(test_file_global):
            print(f"Critical Warning: Global test file '{test_file_global}' not found and dummy creation might have failed. Attempting to proceed with placeholder.")
            dummy_test_samples = 300 # Default test samples
            # Use fixed_original_dimension for placeholder if actual file dimension is unknown
            testing_vectors_loaded_main_global = np.random.rand(dummy_test_samples, fixed_original_dimension).astype(np.float32)
            print(f"Using fallback dummy global testing_vectors shape: {testing_vectors_loaded_main_global.shape}")
        else:
            testing_vectors_loaded_main_global = load_vectors(test_file_global)
            print(f"Loaded actual global testing_vectors shape: {testing_vectors_loaded_main_global.shape}")
            if testing_vectors_loaded_main_global.shape[1] < fixed_original_dimension:
                print(f"Warning: Test data feature dimension ({testing_vectors_loaded_main_global.shape[1]}) is less than fixed_original_dimension ({fixed_original_dimension}). Selection will use all available features from test data.")
    except Exception as e_load_test:
        print(f"Critical Error: Could not load or create global test data '{test_file_global}': {e_load_test}. Exiting.")
        exit()

    num_parallel_workers_pool = mp.cpu_count()
    init_global_pool(num_workers=num_parallel_workers_pool)

    # --- Loop for each training file (Scalability Test Core) ---
    for train_file_current in training_files_to_test:
        current_run_overall_start_time = time.perf_counter()

        # Create dataset identifier for logging (e.g., "300_Fasttext")
        try:
            parts = train_file_current.split('_')
            dataset_size_id = parts[2] # e.g., "300"
            # Verify dataset name from file, though CURRENT_DATASET_NAME is primary for this run
            file_dataset_name_part = parts[3].split('.')[0]
            if file_dataset_name_part != CURRENT_DATASET_NAME:
                print(f"Warning: Filename part '{file_dataset_name_part}' differs from configured '{CURRENT_DATASET_NAME}'.")
            dataset_identifier_for_logging_and_internal_id = f"{dataset_size_id}_{CURRENT_DATASET_NAME}"
        except IndexError:
            # Fallback if filename format is different than "training_vectors_SIZE_NAME.npy"
            dataset_size_id = train_file_current.split('.')[0]
            dataset_identifier_for_logging_and_internal_id = f"{dataset_size_id}_{CURRENT_DATASET_NAME}"


        print(f"\n\n--- Starting Scalability Test Iteration for Training File: {train_file_current} (ID: {dataset_identifier_for_logging_and_internal_id}) ---")

        print(f"Loading training data: {train_file_current}")
        try:
            if not os.path.exists(train_file_current):
                print(f"Critical Warning: Training file '{train_file_current}' not found and dummy creation might have failed. Attempting to proceed with placeholder.")
                try:
                    num_samples_str = train_file_current.split('_')[2]
                    dummy_num_samples = int(num_samples_str)
                except (IndexError, ValueError):
                    dummy_num_samples = 600 # Default if parsing fails
                training_vectors_loaded_main = np.random.rand(dummy_num_samples, fixed_original_dimension).astype(np.float32)
                print(f"Using fallback dummy training_vectors shape: {training_vectors_loaded_main.shape} for {train_file_current}")
            else:
                training_vectors_loaded_main = load_vectors(train_file_current)
                print(f"Loaded actual training_vectors shape: {training_vectors_loaded_main.shape} for {train_file_current}")
                if training_vectors_loaded_main.shape[1] < fixed_original_dimension:
                    print(f"Warning: Training data {train_file_current} feature dimension ({training_vectors_loaded_main.shape[1]}) is less than fixed_original_dimension ({fixed_original_dimension}). Selection will use all available features from training data.")
        except FileNotFoundError as e_fnf:
            print(f"Error: Training file not found - {e_fnf}. Skipping this iteration.")
            continue
        except Exception as e_load:
            print(f"An error occurred during data loading for {train_file_current}: {e_load}. Skipping this iteration.")
            continue

        mem_after_data_load_main = get_memory_usage()
        print(f"Data loading for {train_file_current} complete. Memory usage: {mem_after_data_load_main / (1024**2):.2f} MB")

        print(f"\n--- Experiment Parameter Settings for this run ({CURRENT_DATASET_NAME}) ---")
        print(f"DW-PMAD 'b' value: {b_values_config[0]}")
        print(f"DW-PMAD 'alpha' value: {alpha_values_config[0]}")
        print(f"k-NN 'k' values for accuracy: {k_values_for_knn_config}")
        print(f"Initial dimensions to select from dataset: {dimensions_to_select_config[0]}")
        print(f"Target DR ratio (of selected dimension): {target_ratios_for_dr_config[0]}")

        parameter_combinations_for_run_list = list(itertools.product(
            dimensions_to_select_config,
            target_ratios_for_dr_config,
            b_values_config,
            alpha_values_config
        ))

        current_training_file_performance_data_list = []

        for current_params_tuple_iter in tqdm(parameter_combinations_for_run_list, desc=f"Processing for {train_file_current}"):
            process_parameters(
                current_params_tuple_iter,
                use_dw_pmad_main_execution_flag,
                training_vectors_loaded_main,
                testing_vectors_loaded_main_global,
                k_values_for_knn_config,
                current_training_file_performance_data_list,
                current_dataset_id=dataset_identifier_for_logging_and_internal_id
            )

        # --- Reporting Section (generates outputs per training file) ---
        print(f"\nProcessing results for Excel/CSV export for dataset: {dataset_identifier_for_logging_and_internal_id}...")
        excel_writer_data_frames_dict = {}
        ann_method_names_for_excel_sheets = ["Exact_kNN", "HNSWFlat_Faiss", "IVFPQ_Faiss", "HNSWPQ_Faiss", "IVFOPQ_Faiss"]
        for acc_method_sheet_name in ann_method_names_for_excel_sheets:
            excel_writer_data_frames_dict[acc_method_sheet_name] = []

        dr_method_names_ordered_for_excel = [
            'dw_pmad', 'pca', 'umap', 'isomap', 'kernel_pca', 'mds',
            'RandomProjection', 'FastICA', 'tSNE', 'NMF', 'LLE',
            'FeatureAgglomeration', 'Autoencoder', 'VAE'
        ]

        for run_data_item_excel in current_training_file_performance_data_list:
            base_info_for_row_dict = {}
            base_info_for_row_dict['Dataset_ID'] = run_data_item_excel.get('dataset_id', dataset_identifier_for_logging_and_internal_id)
            base_info_for_row_dict['Dimension_Selected_Config'] = run_data_item_excel['params'][0]
            base_info_for_row_dict['Target_Ratio_DR_Config'] = run_data_item_excel['params'][1]
            if len(run_data_item_excel['params']) > 3:
                base_info_for_row_dict['b_dwpmad_Config'] = run_data_item_excel['params'][2]
                base_info_for_row_dict['alpha_dwpmad_Config'] = run_data_item_excel['params'][3]
            else:
                base_info_for_row_dict['b_dwpmad_Config'] = "N/A"
                base_info_for_row_dict['alpha_dwpmad_Config'] = "N/A"

            base_info_for_row_dict['Orig_Dim_Actual_Selected'] = run_data_item_excel['orig_dim_selected']
            base_info_for_row_dict['Target_Dim_Actual_DR'] = run_data_item_excel['target_dim_final_dr']

            for k_val_iter_excel in k_values_for_knn_config:
                row_for_k_val_excel_base = base_info_for_row_dict.copy()
                row_for_k_val_excel_base['k_Neighbors'] = k_val_iter_excel

                for dr_method_col_name in dr_method_names_ordered_for_excel:
                    dr_time = run_data_item_excel.get('dr_timings', {}).get(dr_method_col_name, np.nan)
                    row_for_k_val_excel_base[f'{dr_method_col_name}_DR_Time'] = dr_time

                for acc_method_sheet_name_iter in ann_method_names_for_excel_sheets:
                    current_row_for_sheet = row_for_k_val_excel_base.copy()
                    for dr_method_col_name in dr_method_names_ordered_for_excel:
                        current_row_for_sheet[f'{dr_method_col_name}_Accuracy'] = np.nan
                        current_row_for_sheet[f'{dr_method_col_name}_Accuracy_Time'] = np.nan
                    if k_val_iter_excel in run_data_item_excel.get('accuracy_results', {}) and \
                            acc_method_sheet_name_iter in run_data_item_excel['accuracy_results'][k_val_iter_excel]:
                        for dr_method_name_found, acc_val_found_excel in run_data_item_excel['accuracy_results'][k_val_iter_excel][acc_method_sheet_name_iter].items():
                            if dr_method_name_found in dr_method_names_ordered_for_excel:
                                current_row_for_sheet[f'{dr_method_name_found}_Accuracy'] = acc_val_found_excel
                                acc_time_val = run_data_item_excel.get('accuracy_times', {}) \
                                    .get(k_val_iter_excel, {}) \
                                    .get(acc_method_sheet_name_iter, {}) \
                                    .get(dr_method_name_found, np.nan)
                                current_row_for_sheet[f'{dr_method_name_found}_Accuracy_Time'] = acc_time_val
                    excel_writer_data_frames_dict[acc_method_sheet_name_iter].append(current_row_for_sheet)

        # Construct output filenames including the dataset identifier and the parameter string
        output_excel_filename_val = f'scalability_ANN_SOTA_results_{dataset_identifier_for_logging_and_internal_id}_{params_string_for_filename}.xlsx'
        output_csv_base_prefix = f'scalability_ANN_SOTA_results_{dataset_identifier_for_logging_and_internal_id}_{params_string_for_filename}'

        cols_first_part_excel = ['Dataset_ID','Dimension_Selected_Config', 'Target_Ratio_DR_Config',
                                 'b_dwpmad_Config', 'alpha_dwpmad_Config',
                                 'Orig_Dim_Actual_Selected', 'Target_Dim_Actual_DR', 'k_Neighbors']
        ordered_dr_metric_cols = []
        for dr_name in dr_method_names_ordered_for_excel:
            ordered_dr_metric_cols.append(f'{dr_name}_Accuracy')
            ordered_dr_metric_cols.append(f'{dr_name}_DR_Time')
            ordered_dr_metric_cols.append(f'{dr_name}_Accuracy_Time')
        final_cols_order_excel = cols_first_part_excel + ordered_dr_metric_cols

        try:
            with pd.ExcelWriter(output_excel_filename_val, engine='openpyxl') as writer_obj_excel:
                for acc_method_sheet_name_write, data_list_for_sheet_write in excel_writer_data_frames_dict.items():
                    if data_list_for_sheet_write:
                        df_for_sheet_write = pd.DataFrame(data_list_for_sheet_write)
                        for col_ensure_excel in final_cols_order_excel:
                            if col_ensure_excel not in df_for_sheet_write.columns:
                                df_for_sheet_write[col_ensure_excel] = np.nan
                        df_for_sheet_write = df_for_sheet_write[final_cols_order_excel]
                        df_for_sheet_write.to_excel(writer_obj_excel, sheet_name=acc_method_sheet_name_write, index=False)
                    else:
                        print(f"No data to write for Excel sheet: {acc_method_sheet_name_write} (Dataset: {dataset_identifier_for_logging_and_internal_id})")
            print(f"Results for dataset {dataset_identifier_for_logging_and_internal_id} exported to Excel: '{output_excel_filename_val}'")
        except ImportError:
            print(f"Module 'openpyxl' not found. Saving results as individual CSV files for dataset {dataset_identifier_for_logging_and_internal_id}.")
            for acc_method_sheet_name_csv, data_list_for_csv_write in excel_writer_data_frames_dict.items():
                if data_list_for_csv_write:
                    df_for_csv_write = pd.DataFrame(data_list_for_csv_write)
                    for col_ensure_csv in final_cols_order_excel:
                        if col_ensure_csv not in df_for_csv_write.columns:
                            df_for_csv_write[col_ensure_csv] = np.nan
                    df_for_csv_write = df_for_csv_write[final_cols_order_excel]
                    csv_filename = f"{output_csv_base_prefix}_{acc_method_sheet_name_csv}.csv"
                    df_for_csv_write.to_csv(csv_filename, index=False)
                    print(f"Results for {acc_method_sheet_name_csv} (Dataset: {dataset_identifier_for_logging_and_internal_id}) exported to CSV: '{csv_filename}'")
                else:
                    print(f"No data to write for CSV: {acc_method_sheet_name_csv} (Dataset: {dataset_identifier_for_logging_and_internal_id})")

        print(f"Generating performance report for dataset: {dataset_identifier_for_logging_and_internal_id}...")
        output_report_filename_val = f'performance_report_SOTA_ANN_{dataset_identifier_for_logging_and_internal_id}_{params_string_for_filename}.txt'
        current_run_total_time = time.perf_counter() - current_run_overall_start_time

        aggregated_dr_method_times_report = {dr_name: 0.0 for dr_name in dr_method_names_ordered_for_excel}
        aggregated_accuracy_method_times_report = {
            acc_m_name: {dr_name: 0.0 for dr_name in dr_method_names_ordered_for_excel}
            for acc_m_name in ann_method_names_for_excel_sheets
        }

        for run_data_item_perf_report in current_training_file_performance_data_list:
            for dr_method_name_perf, time_val_perf_dr in run_data_item_perf_report.get('dr_timings', {}).items():
                if dr_method_name_perf in aggregated_dr_method_times_report:
                    aggregated_dr_method_times_report[dr_method_name_perf] += (time_val_perf_dr if pd.notna(time_val_perf_dr) else 0.0)
            for k_val_perf_report, acc_methods_data_perf_report in run_data_item_perf_report.get('accuracy_times', {}).items():
                for acc_method_name_perf, dr_times_data_perf_report in acc_methods_data_perf_report.items():
                    if acc_method_name_perf in aggregated_accuracy_method_times_report:
                        for dr_method_name_perf_inner, time_acc_val_perf_report in dr_times_data_perf_report.items():
                            if dr_method_name_perf_inner in aggregated_accuracy_method_times_report[acc_method_name_perf]:
                                current_total_time_for_acc = aggregated_accuracy_method_times_report[acc_method_name_perf].get(dr_method_name_perf_inner, 0.0)
                                aggregated_accuracy_method_times_report[acc_method_name_perf][dr_method_name_perf_inner] = \
                                    current_total_time_for_acc + (time_acc_val_perf_report if pd.notna(time_acc_val_perf_report) else 0.0)
        with open(output_report_filename_val, 'w') as f_report_obj:
            f_report_obj.write(f"--- SOTA ANN Performance Report ({CURRENT_DATASET_NAME} Dataset: {dataset_identifier_for_logging_and_internal_id}, Params: {params_string_for_filename}) ---\n\n")
            f_report_obj.write(f"Test Iteration for Training File: {train_file_current}\n")
            f_report_obj.write(f"Total Time for this Test Iteration: {current_run_total_time:.4f}s\n")
            f_report_obj.write(f"Initial Memory (at script start): {initial_memory_at_script_start / (1024**2):.2f} MB\n")
            f_report_obj.write(f"Memory (after loading {train_file_current}): {mem_after_data_load_main / (1024**2):.2f} MB\n\n")
            f_report_obj.write("--- Aggregated Timings for this Dataset ---\n")
            f_report_obj.write("Dimensionality Reduction Methods (Total Time for this dataset run):\n")
            for dr_method_name_agg_report, total_t_agg_report in aggregated_dr_method_times_report.items():
                f_report_obj.write(f"  - {dr_method_name_agg_report}: {total_t_agg_report:.4f}s\n")
            f_report_obj.write("\nAccuracy Checking Methods (Total Time, Summed over all k-values for this dataset run, per DR method):\n")
            for acc_method_name_agg_report, dr_times_data_agg_report in aggregated_accuracy_method_times_report.items():
                f_report_obj.write(f"  - Accuracy Method: {acc_method_name_agg_report}\n")
                for dr_method_name_agg_inner_report, total_t_acc_agg_report in dr_times_data_agg_report.items():
                    f_report_obj.write(f"    - On {dr_method_name_agg_inner_report} reduced data: {total_t_acc_agg_report:.4f}s\n")
            f_report_obj.write("\n--- Detailed Timings and Memory for this Dataset Run ---\n")
            for i, run_data_item_detail_report in enumerate(current_training_file_performance_data_list):
                params_cfg = run_data_item_detail_report['params']
                params_cfg_str = f"DimSel:{params_cfg[0]}, TgtRatio:{params_cfg[1]}"
                if len(params_cfg) > 3:
                    params_cfg_str += f", b:{params_cfg[2]}, alpha:{params_cfg[3]}"
                f_report_obj.write(f"\nRun Details (Params Config: {params_cfg_str}, "
                                   f"Actual Orig Dim: {run_data_item_detail_report['orig_dim_selected']}, "
                                   f"Final DR Target Dim: {run_data_item_detail_report['target_dim_final_dr']})\n")
                f_report_obj.write(f"  Time - Standardization: {run_data_item_detail_report.get('time_standardization', np.nan):.4f}s\n")
                f_report_obj.write(f"  Memory - After Standardization: {run_data_item_detail_report.get('mem_after_standardization', 0) / (1024**2):.2f} MB\n")
                f_report_obj.write("  DR Method Timings:\n")
                for dr_method_name_detail_report, t_val_detail_report in run_data_item_detail_report.get('dr_timings',{}).items():
                    if dr_method_name_detail_report in dr_method_names_ordered_for_excel:
                        f_report_obj.write(f"    {dr_method_name_detail_report}: {t_val_detail_report:.4f}s\n")
                f_report_obj.write("  Memory - After Each DR Method:\n")
                for dr_method_name_detail_mem_report, m_val_detail_report in run_data_item_detail_report.get('dr_memory_after',{}).items():
                    if dr_method_name_detail_mem_report in dr_method_names_ordered_for_excel:
                        f_report_obj.write(f"    After {dr_method_name_detail_mem_report}: {m_val_detail_report / (1024**2):.2f} MB\n")
                f_report_obj.write("  Accuracy Calculation Times (per k, per accuracy method, per DR method):\n")
                for k_val_detail_report, acc_methods_data_detail_report in run_data_item_detail_report.get('accuracy_times', {}).items():
                    f_report_obj.write(f"    For k={k_val_detail_report}:\n")
                    for acc_method_name_detail_report, dr_times_data_detail_report in acc_methods_data_detail_report.items():
                        f_report_obj.write(f"      Accuracy Method: {acc_method_name_detail_report}\n")
                        for dr_method_name_detail_inner_report, t_acc_val_detail_report in dr_times_data_detail_report.items():
                            if dr_method_name_detail_inner_report in dr_method_names_ordered_for_excel:
                                f_report_obj.write(f"        Time on {dr_method_name_detail_inner_report} reduced data: {t_acc_val_detail_report:.4f}s\n")
                f_report_obj.write(f"  Memory - After All Accuracy Calcs for this run: {run_data_item_detail_report.get('mem_after_accuracy', 0) / (1024**2):.2f} MB\n")
                f_report_obj.write(f"  Peak Memory Observed During this run: {run_data_item_detail_report.get('peak_memory_in_run', 0) / (1024**2):.2f} MB\n")
        print(f"Performance report for dataset {dataset_identifier_for_logging_and_internal_id} generated: '{output_report_filename_val}'")
        print(f"--- Finished Scalability Test Iteration for: {train_file_current} ---")

    if global_pool:
        global_pool.close()
        global_pool.join()
        print("\nGlobal multiprocessing pool closed.")

    overall_end_time_script_exec_final = time.perf_counter()
    total_execution_time_for_script_overall = overall_end_time_script_exec_final - overall_start_time_script_exec
    print(f"\nTotal script execution time for all datasets: {total_execution_time_for_script_overall:.4f}s")
