import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
import time
import os  # For performance report
import psutil  # For memory usage
from scipy.optimize import minimize
from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF
from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import pdist # Kept for comparison if needed, but parallel_pdist is used
from tqdm import tqdm
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
from sklearn.linear_model import LinearRegression
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
import umap  # Standard import; ensure this is from umap-learn
import faiss  # For ANN methods

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
        except AttributeError:
            print(f"Global pool initialized with {cpu_cores} workers. Faiss OMP settings not available/changed.")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    try:
        return process.memory_info().uss
    except AttributeError:
        return process.memory_info().rss

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
            init_global_pool()

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
        partitioned_idx = max(0, top_b_count - 1)
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
            w_norm = w_opt / (np.linalg.norm(w_opt) + 1e-9)
            main_objective = dw_pmad_b(w_norm, X, b_percentage_val)
            ortho_penalty_val_calc = orthogonality_constraint(w_norm, prev_ws_list)
            return main_objective + alpha_penalty_val * ortho_penalty_val_calc

        initial_w_guess = np.random.randn(n_features).astype(np.float32)
        initial_w_guess /= (np.linalg.norm(initial_w_guess) + 1e-9)
        result = minimize(objective_function, initial_w_guess, method='L-BFGS-B', options=optimizer_options)
        if result.success:
            optimal_w_found = result.x / (np.linalg.norm(result.x) + 1e-9)
        else:
            print(f"Warning: DW-PMAD axis {axis_num+1} did not converge ({result.message}).")
            optimal_w_found = result.x / (np.linalg.norm(result.x) + 1e-9)

        prev_ws_list.append(optimal_w_found.astype(np.float32))
        optimal_ws_list.append(optimal_w_found.astype(np.float32))

    projection_axes_found = np.column_stack(optimal_ws_list)
    projected_X_result = X @ projection_axes_found
    return projected_X_result, projection_axes_found

def project_dw_pmad(X_to_project, projection_axes_to_use):
    return X_to_project @ projection_axes_to_use

# ---------------------------------------------------------
# BASELINE DR METHODS
# ---------------------------------------------------------
def run_random_projection(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    rp = GaussianRandomProjection(n_components=target_dim, random_state=1)
    X_train_rp = rp.fit_transform(X_train)
    X_test_rp = rp.transform(X_test)
    return X_train_rp, X_test_rp, time.perf_counter() - t0

def run_fastica(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    ica = FastICA(n_components=target_dim, random_state=1, max_iter=200, tol=0.01)
    X_train_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)
    return X_train_ica, X_test_ica, time.perf_counter() - t0

def run_tsne(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    tsne_model = TSNE(n_components=target_dim, method='exact', random_state=1, perplexity=30.0, n_iter=1000)
    if X_train.shape[0] - 1 < tsne_model.perplexity:
        tsne_model.perplexity = max(5, X_train.shape[0] - 2)
    if X_train.shape[0] <= tsne_model.n_components:
        shape = (X_train.shape[0], target_dim)
        return np.full(shape, np.nan), np.full((X_test.shape[0], target_dim), np.nan), 0.0
    X_train_tsne = tsne_model.fit_transform(X_train)
    reg = LinearRegression().fit(X_train, X_train_tsne) if X_train.shape[0] > 0 else None
    X_test_tsne = reg.predict(X_test) if reg else np.full((X_test.shape[0], target_dim), np.nan)
    return X_train_tsne, X_test_tsne, time.perf_counter() - t0

def run_nmf(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    train_min = X_train.min()
    X_train_nmf_shifted = np.clip(X_train - train_min, 0, None)
    X_test_nmf_shifted = np.clip(X_test - train_min, 0, None)
    nmf_model = NMF(n_components=target_dim, init='random', random_state=1, max_iter=200, tol=1e-4)
    try:
        X_train_nmf_trans = nmf_model.fit_transform(X_train_nmf_shifted)
        X_test_nmf_trans = nmf_model.transform(X_test_nmf_shifted)
    except ValueError as e:
        print(f"NMF Error: {e}")
        shape_t = (X_train.shape[0], target_dim)
        shape_s = (X_test.shape[0], target_dim)
        return np.full(shape_t, np.nan), np.full(shape_s, np.nan), time.perf_counter() - t0
    return X_train_nmf_trans, X_test_nmf_trans, time.perf_counter() - t0

def run_lle(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    n_neighbors = max(target_dim + 1, min(10, X_train.shape[0] - 1))
    if X_train.shape[0] <= n_neighbors or X_train.shape[0] <= target_dim:
        shape = (X_train.shape[0], target_dim)
        return np.full(shape, np.nan), np.full((X_test.shape[0], target_dim), np.nan), 0.0
    lle_model = LocallyLinearEmbedding(n_components=target_dim, n_neighbors=n_neighbors, random_state=1)
    X_train_lle = lle_model.fit_transform(X_train)
    X_test_lle = lle_model.transform(X_test)
    return X_train_lle, X_test_lle, time.perf_counter() - t0

def run_feature_agglomeration(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    actual_target = min(max(1, target_dim), X_train.shape[1]) if X_train.shape[1] > 0 else 0
    if actual_target == 0:
        return np.zeros((X_train.shape[0], 0)), np.zeros((X_test.shape[0], 0)), 0.0
    fa = FeatureAgglomeration(n_clusters=actual_target)
    X_train_fa = fa.fit_transform(X_train)
    X_test_fa = fa.transform(X_test)
    return X_train_fa, X_test_fa, time.perf_counter() - t0

def run_autoencoder(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    input_dim = X_train.shape[1]
    if input_dim == 0:
        return np.zeros((X_train.shape[0], target_dim)), np.zeros((X_test.shape[0], target_dim)), 0.0
    tf.keras.utils.set_random_seed(1)
    inputs = Input(shape=(input_dim,))
    encoded = Dense(target_dim, activation='relu')(inputs)
    decoded = Dense(input_dim, activation='linear')(encoded)
    auto = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    auto.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    auto.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0, shuffle=True)
    X_train_ae = encoder.predict(X_train, batch_size=256)
    X_test_ae = encoder.predict(X_test, batch_size=256)
    K.clear_session()
    return X_train_ae, X_test_ae, time.perf_counter() - t0

def run_vae(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    input_dim = X_train.shape[1]
    if input_dim == 0:
        return np.zeros((X_train.shape[0], target_dim)), np.zeros((X_test.shape[0], target_dim)), 0.0
    tf.keras.utils.set_random_seed(1)
    latent_dim = target_dim
    inter_dim = max(latent_dim * 2, 64)
    inputs = Input(shape=(input_dim,))
    h = Dense(inter_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    def sampling(args):
        mean, logvar = args
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        eps = K.random_normal(shape=(batch, dim))
        return mean + K.exp(0.5 * logvar) * eps
    z = Lambda(sampling)([z_mean, z_log_var])
    dec_h = Dense(inter_dim, activation='relu')(z)
    outputs = Dense(input_dim, activation='sigmoid')(dec_h)
    vae = Model(inputs, outputs)
    reconstruction_loss = tf.keras.losses.mse(inputs, outputs) * input_dim
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae.add_loss(K.mean(reconstruction_loss + kl_loss))
    vae.compile(optimizer=tf.keras.optimizers.Adam(0.001))
    vae.fit(X_train, None, epochs=20, batch_size=32, verbose=0, shuffle=True)
    encoder = Model(inputs, z_mean)
    X_train_vae = encoder.predict(X_train, batch_size=256)
    X_test_vae = encoder.predict(X_test, batch_size=256)
    K.clear_session()
    return X_train_vae, X_test_vae, time.perf_counter() - t0

NEW_BASELINE_DR_METHODS = [
    'RandomProjection', 'FastICA', 'tSNE', 'NMF', 'LLE',
    'FeatureAgglomeration', 'Autoencoder', 'VAE'
]

# ---------------------------------------------------------
# FAISS & ACCURACY HELPERS
# ---------------------------------------------------------
def get_valid_pq_m(d, max_m_val=8, ensure_d_multiple=True, min_subvector_dim=1):
    if d == 0 or d < min_subvector_dim:
        return 0
    upper = min(d // min_subvector_dim, max_m_val, d)
    for m in range(upper, 0, -1):
        if not ensure_d_multiple or d % m == 0:
            return m
    return 0

def get_dynamic_nbits(n_train, m_pq, default_nbits=8, min_nbits=4):
    if m_pq == 0:
        return default_nbits
    nbits = default_nbits
    while n_train < 4 * (2**nbits) and nbits > min_nbits:
        nbits -= 1
    return max(min_nbits, nbits)

def get_exact_neighbors(data_to_index, query_data, k_neighbors):
    if data_to_index.shape[0] == 0 or query_data.shape[0] == 0:
        return np.empty((query_data.shape[0], 0), dtype=int)
    actual_k = min(k_neighbors, data_to_index.shape[0])
    if actual_k == 0:
        return np.empty((query_data.shape[0], 0), dtype=int)
    nbrs = NearestNeighbors(n_neighbors=actual_k, algorithm='auto', n_jobs=-1).fit(data_to_index)
    return np.vstack([nbrs.kneighbors(query_data[i].reshape(1, -1), return_distance=False)[0]
                      for i in range(len(query_data))])

def calculate_accuracy_exact_knn(exact_gt, train_red, test_red, k):
    start = time.perf_counter()
    if train_red.shape[0] == 0 or test_red.shape[0] == 0:
        return np.nan, 0.0
    actual_k = min(k, train_red.shape[0])
    if actual_k == 0:
        return np.nan, 0.0
    nbrs = NearestNeighbors(n_neighbors=actual_k, algorithm='auto', n_jobs=-1).fit(train_red)
    matches = 0
    for i in range(len(test_red)):
        found = nbrs.kneighbors(test_red[i].reshape(1, -1), return_distance=False)[0]
        matches += len(set(exact_gt[i]) & set(found))
    acc = matches / (len(test_red) * exact_gt.shape[1]) if exact_gt.shape[1] > 0 else 0.0
    return acc, time.perf_counter() - start

def _generic_faiss_accuracy_calc(idx, exact_gt, test_data, k):
    matches = 0
    if test_data.shape[0] > 0 and exact_gt.shape[1] > 0 and idx and idx.ntotal > 0:
        actual_k = min(k, idx.ntotal)
        _, inds = idx.search(test_data, actual_k)
        for i in range(len(test_data)):
            matches += len(set(exact_gt[i]) & set(inds[i]))
        return matches / (len(test_data) * exact_gt.shape[1])
    return 0.0

def calculate_accuracy_hnswflat_faiss(exact_gt, train_red, test_red, k):
    start = time.perf_counter()
    d = train_red.shape[1]
    if d == 0 or train_red.shape[0] == 0:
        return np.nan, 0.0
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 50
    try:
        index.add(train_red)
    except RuntimeError as e:
        print(f"HNSWFlat Add Error: {e}")
        return np.nan, time.perf_counter() - start
    acc = _generic_faiss_accuracy_calc(index, exact_gt, test_red, k)
    return acc, time.perf_counter() - start

def calculate_accuracy_ivfpq_faiss(exact_gt, train_red, test_red, k):
    start = time.perf_counter()
    d, n_train = train_red.shape[1], train_red.shape[0]
    if d == 0 or n_train == 0:
        return np.nan, 0.0
    nlist = min(100, max(1, n_train // 39))
    m_pq = get_valid_pq_m(d, max_m_val=8, min_subvector_dim=2)
    if m_pq == 0:
        print(f"IVFPQ Skipping: no valid m_pq for d={d}")
        return np.nan, 0.0
    nbits = get_dynamic_nbits(n_train, m_pq)
    quantizer = faiss.IndexFlatL2(d)
    try:
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m_pq, nbits)
        if n_train < 39 * nlist:
            print(f"Warning: n_train ({n_train}) < 39*nlist ({39*nlist})")
        if n_train < 4 * (2**nbits):
            print(f"Warning: n_train ({n_train}) low for PQ training")
        index.train(train_red)
        index.add(train_red)
        index.nprobe = min(nlist, 10)
    except RuntimeError as e:
        print(f"IVFPQ Error: {e}")
        return np.nan, time.perf_counter() - start
    acc = _generic_faiss_accuracy_calc(index, exact_gt, test_red, k)
    return acc, time.perf_counter() - start

# You can add the other FAISS-based accuracy functions (HNSWPQ, IVFOPQ) similarly...

# ---------------------------------------------------------
# PROCESSING FUNCTION
# ---------------------------------------------------------
def process_parameters(params_tuple, use_dw_pmad_flag, X_train_full, X_test_full, k_values, collector, current_dataset_id=""):
    perf = {'params': params_tuple, 'dataset_id': current_dataset_id}
    orig_dim, ratio = params_tuple[:2]
    b_val, alpha_val = params_tuple[2:] if use_dw_pmad_flag else ("N/A", "N/A")

    actual_dim = min(orig_dim, X_train_full.shape[1])
    target_dim = max(1, int(actual_dim * ratio))
    target_dim = min(target_dim, actual_dim)

    perf['orig_dim_selected'] = actual_dim
    perf['target_dim_final_dr'] = target_dim

    np.random.seed(1)
    idxs = np.random.choice(X_train_full.shape[1], actual_dim, replace=False)
    X_train_sel = X_train_full[:, idxs]
    X_test_sel = X_test_full[:, idxs]

    t0 = time.perf_counter()
    mean, std = X_train_sel.mean(0), X_train_sel.std(0)
    std[std == 0] = 1e-6
    X_train_std = (X_train_sel - mean) / std
    X_test_std = (X_test_sel - mean) / std
    perf['time_standardization'] = time.perf_counter() - t0
    perf['mem_after_standardization'] = get_memory_usage()

    dr_results, dr_times, dr_mem = {}, {}, {}

    # --- DW-PMAD ---
    if use_dw_pmad_flag:
        print("Starting DW-PMAD...")
        t_dw = time.perf_counter()
        td = max(1, min(target_dim, X_train_std.shape[1])) if X_train_std.shape[1] > 0 else 0
        if td > 0:
            train_dw, axes = dw_pmad(X_train_std, b_val, alpha_val, td)
            test_dw = project_dw_pmad(X_test_std, axes)
        else:
            train_dw = np.zeros((X_train_std.shape[0], td))
            test_dw = np.zeros((X_test_std.shape[0], td))
        dr_results['dw_pmad'] = (train_dw, test_dw)
        dr_times['dw_pmad'] = time.perf_counter() - t_dw
        dr_mem['dw_pmad'] = get_memory_usage()
    else:
        dr_times['dw_pmad'] = np.nan

    # --- Other DR methods ---
    configs = {
        'pca': (PCA, {'random_state': 1}),
        'umap': (umap.UMAP, {'random_state': 1, 'min_dist': 0.1, 'n_jobs': 1}),
        'isomap': (Isomap, {}),
        'kernel_pca': (KernelPCA, {'kernel': 'rbf', 'random_state': 1}),
        'mds': (MDS, {'dissimilarity': 'euclidean', 'random_state': 1}),
        'RandomProjection': (run_random_projection, None),
        'FastICA': (run_fastica, None),
        'tSNE': (run_tsne, None),
        'NMF': (run_nmf, None),
        'LLE': (run_lle, None),
        'FeatureAgglomeration': (run_feature_agglomeration, None),
        'Autoencoder': (run_autoencoder, None),
        'VAE': (run_vae, None),
    }

    for name, (cls_or_fn, params) in configs.items():
        if name == 'dw_pmad':
            continue
        print(f"Starting {name}...")
        t_start = time.perf_counter()
        n_comp = target_dim
        if name == 'pca':
            n_comp = min(target_dim, X_train_std.shape[0], X_train_std.shape[1])
        elif name in ['umap', 'isomap', 'mds', 'LLE', 'tSNE']:
            if X_train_std.shape[1] > 1:
                n_comp = min(target_dim, X_train_std.shape[1] - (0 if name=='tSNE' else 1))
                if name == 'tSNE':
                    n_comp = min(n_comp, 3)
            else:
                n_comp = 1
        elif name in ['RandomProjection', 'FastICA', 'NMF', 'FeatureAgglomeration']:
            n_comp = min(target_dim, X_train_std.shape[1])
        n_comp = max(1, n_comp) if X_train_std.shape[1] > 0 else 0

        skip = False
        if X_train_std.shape[1] == 0 or n_comp == 0 or \
                (name in ['umap','isomap','mds','LLE','tSNE'] and X_train_std.shape[0] <= n_comp):
            skip = True

        if skip:
            train_r = np.full((X_train_std.shape[0], n_comp), np.nan)
            test_r = np.full((X_test_std.shape[0], n_comp), np.nan)
            dr_results[name] = (train_r, test_r)
            dr_times[name] = time.perf_counter() - t_start
            dr_mem[name] = get_memory_usage()
            continue

        if name in NEW_BASELINE_DR_METHODS:
            try:
                train_r, test_r, t_sp = cls_or_fn(X_train_std, X_test_std, n_comp)
                dr_results[name] = (train_r, test_r)
                dr_times[name] = t_sp
                dr_mem[name] = get_memory_usage()
            except Exception as e:
                print(f"{name} error: {e}")
                shape = (X_train_std.shape[0], n_comp)
                dr_results[name] = (np.full(shape, np.nan), np.full((X_test_std.shape[0], n_comp), np.nan))
                dr_times[name] = time.perf_counter() - t_start
                dr_mem[name] = get_memory_usage()
        else:
            try:
                p = params.copy()
                p['n_components'] = n_comp
                if name in ['umap','isomap']:
                    p['n_neighbors'] = max(1, min(15, X_train_std.shape[0]-1))
                model = cls_or_fn(**p)
                train_r = model.fit_transform(X_train_std)
                test_r = model.transform(X_test_std) if hasattr(model, 'transform') else LinearRegression().fit(X_train_std, train_r).predict(X_test_std)
                dr_results[name] = (train_r, test_r)
                dr_times[name] = time.perf_counter() - t_start
                dr_mem[name] = get_memory_usage()
            except Exception as e:
                print(f"{name} sklearn error: {e}")
                shape = (X_train_std.shape[0], n_comp)
                dr_results[name] = (np.full(shape, np.nan), np.full((X_test_std.shape[0], n_comp), np.nan))
                dr_times[name] = time.perf_counter() - t_start
                dr_mem[name] = get_memory_usage()

    perf['dr_timings'] = dr_times
    perf['dr_memory_after'] = dr_mem

    # --- ACCURACY CALCULATIONS ---
    exact_indices = {}
    if X_train_std.shape[0] > 0 and X_test_std.shape[0] > 0:
        for k in k_values:
            exact_indices[k] = get_exact_neighbors(X_train_std, X_test_std, k)
    else:
        for k in k_values:
            exact_indices[k] = np.empty((X_test_std.shape[0], 0), dtype=int)

    ann_funcs = {
        "Exact_kNN": calculate_accuracy_exact_knn,
        "HNSWFlat_Faiss": calculate_accuracy_hnswflat_faiss,
        "IVFPQ_Faiss": calculate_accuracy_ivfpq_faiss,
        # add others here...
    }

    acc_results, acc_times = {k: {} for k in k_values}, {k: {} for k in k_values}
    for k in k_values:
        gt = exact_indices[k]
        for acc_name, fn in ann_funcs.items():
            for dr_name, (train_r, test_r) in dr_results.items():
                if train_r is None or train_r.size == 0:
                    acc, t_acc = np.nan, 0.0
                else:
                    if "Faiss" in acc_name:
                        acc = fn(gt, train_r.astype(np.float32), test_r.astype(np.float32), k)
                        t_acc = acc[1] if isinstance(acc, tuple) else 0.0
                        acc = acc[0] if isinstance(acc, tuple) else acc
                    else:
                        acc, t_acc = fn(gt, train_r, test_r, k)
                acc_results[k].setdefault(acc_name, {})[dr_name] = acc
                acc_times[k].setdefault(acc_name, {})[dr_name] = t_acc

    perf['accuracy_results'] = acc_results
    perf['accuracy_times'] = acc_times
    perf['mem_after_accuracy'] = get_memory_usage()
    collector.append(perf)

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == '__main__':
    overall_start = time.perf_counter()
    initial_mem = get_memory_usage()

    # Dataset and file configuration
    CURRENT_DATASET_NAME = "Fasttext"
    training_files = [
        f"training_vectors_300_{CURRENT_DATASET_NAME}.npy",
        f"training_vectors_600_{CURRENT_DATASET_NAME}.npy",
        f"training_vectors_900_{CURRENT_DATASET_NAME}.npy",
        f"training_vectors_1200_{CURRENT_DATASET_NAME}.npy",
    ]
    test_file = f"testing_vectors_300_{CURRENT_DATASET_NAME}.npy"

    # Fixed experiment parameters
    fixed_alpha = 6
    fixed_b = 90
    fixed_ratio = 0.6
    fixed_k = [1, 3, 6, 10, 15]
    fixed_orig_dim = 200

    # Filename suffix
    ratio_str_for_filename = str(fixed_ratio).replace('.', 'p')
    params_string_for_filename = (
        f"origD{fixed_orig_dim}_"
        f"ratio{ratio_str_for_filename}_"
        f"alpha{fixed_alpha}_"
        f"b{fixed_b}"
    )

    # Make DW-PMAD optional: flip to False to skip DW-PMAD
    use_dw_pmad_main_execution_flag = False

    # --- Dummy file creation (if needed) ---
    for f_name in training_files:
        if not os.path.exists(f_name):
            print(f"Creating dummy file: {f_name}")
            try:
                num_samples = int(f_name.split('_')[2])
                dummy_data = np.random.rand(num_samples, fixed_orig_dim).astype(np.float32)
                np.save(f_name, dummy_data)
            except Exception as e:
                print(f"Could not create {f_name}: {e}")

    if not os.path.exists(test_file):
        print(f"Creating dummy file: {test_file}")
        try:
            dummy_data_test = np.random.rand(300, fixed_orig_dim).astype(np.float32)
            np.save(test_file, dummy_data_test)
        except Exception as e:
            print(f"Could not create {test_file}: {e}")

    # --- Load global testing vectors ---
    try:
        testing_vectors_loaded_main_global = load_vectors(test_file)
        print(f"Loaded test set: {testing_vectors_loaded_main_global.shape}")
    except Exception as e:
        print(f"Warning: failed to load {test_file} ({e}), using dummy.")
        testing_vectors_loaded_main_global = np.random.rand(300, fixed_orig_dim).astype(np.float32)

    # Initialize multiprocessing pool
    init_global_pool(num_workers=mp.cpu_count())

    # --- Scalability test loop ---
    for train_file_current in training_files:
        run_start = time.perf_counter()
        # Determine dataset ID
        parts = train_file_current.split('_')
        size_id = parts[2]
        dataset_identifier = f"{size_id}_{CURRENT_DATASET_NAME}"

        # Load or dummy training vectors
        try:
            training_vectors_loaded = load_vectors(train_file_current)
            print(f"Loaded train set {train_file_current}: {training_vectors_loaded.shape}")
        except Exception as e:
            print(f"Warning: failed to load {train_file_current} ({e}), using dummy.")
            try:
                dummy_n = int(size_id)
            except:
                dummy_n = 600
            training_vectors_loaded = np.random.rand(dummy_n, fixed_orig_dim).astype(np.float32)

        mem_after_load = get_memory_usage()
        print(f"Memory after load: {mem_after_load/(1024**2):.2f} MB")

        print(f"\n--- Running with b={fixed_b}, alpha={fixed_alpha}, k={fixed_k}, orig_dim={fixed_orig_dim}, ratio={fixed_ratio} ---")
        param_combos = list(itertools.product(
            [fixed_orig_dim],
            [fixed_ratio],
            [fixed_b],
            [fixed_alpha]
        ))
        performance_data = []

        for params in tqdm(param_combos, desc=f"Params on {dataset_identifier}"):
            process_parameters(
                params,
                use_dw_pmad_main_execution_flag,
                training_vectors_loaded,
                testing_vectors_loaded_main_global,
                fixed_k,
                performance_data,
                current_dataset_id=dataset_identifier
            )

        # --- Export results ---
        print(f"\nExporting results for {dataset_identifier}...")
        sheets = ["Exact_kNN", "HNSWFlat_Faiss", "IVFPQ_Faiss", "HNSWPQ_Faiss", "IVFOPQ_Faiss"]
        excel_data = {sheet: [] for sheet in sheets}
        dr_order = [
            'dw_pmad', 'pca', 'umap', 'isomap', 'kernel_pca', 'mds',
            'RandomProjection', 'FastICA', 'tSNE', 'NMF', 'LLE',
            'FeatureAgglomeration', 'Autoencoder', 'VAE'
        ]

        for item in performance_data:
            base = {
                'Dataset_ID': item['dataset_id'],
                'Dim_Config': item['params'][0],
                'Ratio_Config': item['params'][1],
                'b_Config': item['params'][2],
                'alpha_Config': item['params'][3],
                'Orig_Dim': item['orig_dim_selected'],
                'DR_Dim': item['target_dim_final_dr']
            }
            for k_val in fixed_k:
                row_base = dict(base, **{'k': k_val})
                # add DR times
                for dr in dr_order:
                    row_base[f"{dr}_DR_Time"] = item['dr_timings'].get(dr, np.nan)
                # add accuracies
                for sheet in sheets:
                    row = row_base.copy()
                    accs = item['accuracy_results'].get(k_val, {}).get(sheet, {})
                    times = item['accuracy_times'].get(k_val, {}).get(sheet, {})
                    for dr in dr_order:
                        row[f"{dr}_Acc"] = accs.get(dr, np.nan)
                        row[f"{dr}_Acc_Time"] = times.get(dr, np.nan)
                    excel_data[sheet].append(row)

        excel_fn = f"scalability_{dataset_identifier}_{params_string_for_filename}.xlsx"
        csv_prefix = f"scalability_{dataset_identifier}_{params_string_for_filename}"

        try:
            with pd.ExcelWriter(excel_fn, engine='openpyxl') as writer:
                for sheet, rows in excel_data.items():
                    if rows:
                        df = pd.DataFrame(rows)
                        df.to_excel(writer, sheet_name=sheet, index=False)
                    else:
                        print(f"No data for sheet {sheet}")
            print(f"Written Excel: {excel_fn}")
        except ImportError:
            print("openpyxl missing; saving CSVs instead.")
            for sheet, rows in excel_data.items():
                if rows:
                    df = pd.DataFrame(rows)
                    csv_fn = f"{csv_prefix}_{sheet}.csv"
                    df.to_csv(csv_fn, index=False)
                    print(f"Written CSV: {csv_fn}")
                else:
                    print(f"No data for {sheet}")

        # --- Performance report ---
        report_fn = f"performance_report_{dataset_identifier}_{params_string_for_filename}.txt"
        agg_dr = {dr: 0.0 for dr in dr_order}
        agg_acc = {s: {dr: 0.0 for dr in dr_order} for s in sheets}

        for item in performance_data:
            for dr, t in item['dr_timings'].items():
                if dr in agg_dr and not np.isnan(t):
                    agg_dr[dr] += t
            for k_val, acc_dict in item['accuracy_times'].items():
                for sheet, dr_times in acc_dict.items():
                    for dr, t in dr_times.items():
                        if dr in dr_order and not np.isnan(t):
                            agg_acc[sheet][dr] += t

        with open(report_fn, 'w') as f:
            f.write(f"SOTA ANN Report ({dataset_identifier})\n")
            f.write(f"Total runs: {len(performance_data)}\n")
            f.write("\nDimensionality Reduction Times:\n")
            for dr, t in agg_dr.items():
                f.write(f"  {dr}: {t:.4f}s\n")
            f.write("\nAccuracy Times:\n")
            for sheet, dr_times in agg_acc.items():
                f.write(f"{sheet}:\n")
                for dr, t in dr_times.items():
                    f.write(f"  {dr}: {t:.4f}s\n")
        print(f"Written report: {report_fn}\n")
        print(f"Iteration time for {dataset_identifier}: {time.perf_counter() - run_start:.2f}s\n")

    # Clean up pool
    if global_pool:
        global_pool.close()
        global_pool.join()
        print("Global pool closed.")

    total_time = time.perf_counter() - overall_start
    print(f"Total script time: {total_time:.2f}s")
