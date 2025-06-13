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
# from scipy.spatial.distance import pdist  # Kept for comparison if needed, but parallel_pdist is used
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
        return process.memory_info().uss  # Unique Set Size
    except AttributeError:
        return process.memory_info().rss  # Resident Set Size

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
# (Defined but will be skipped since flag=False)
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
            ortho_penalty = orthogonality_constraint(w_norm, prev_ws_list)
            return main_objective + alpha_penalty_val * ortho_penalty

        initial_w = np.random.randn(n_features).astype(np.float32)
        initial_w /= (np.linalg.norm(initial_w) + 1e-9)
        result = minimize(objective_function, initial_w, method='L-BFGS-B', options=optimizer_options)
        if result.success:
            w_final = result.x / (np.linalg.norm(result.x) + 1e-9)
        else:
            print(f"Warning: DW-PMAD axis {axis_num+1} did not converge ({result.message}).")
            w_final = result.x / (np.linalg.norm(result.x) + 1e-9)
        prev_ws_list.append(w_final.astype(np.float32))
        optimal_ws_list.append(w_final.astype(np.float32))

    axes = np.column_stack(optimal_ws_list)
    proj = X @ axes
    return proj, axes

def project_dw_pmad(X_to_project, axes):
    return X_to_project @ axes

# ---------------------------------------------------------
# Baseline DR methods
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
    model = TSNE(n_components=target_dim, method='exact', random_state=1, perplexity=30.0, n_iter=1000)
    if X_train.shape[0] - 1 < model.perplexity:
        model.perplexity = max(5, X_train.shape[0] - 2)
    if X_train.shape[0] <= model.n_components:
        return np.full((X_train.shape[0], target_dim), np.nan), np.full((X_test.shape[0], target_dim), np.nan), 0.0
    X_tr = model.fit_transform(X_train)
    reg = LinearRegression().fit(X_train, X_tr)
    X_te = reg.predict(X_test)
    return X_tr, X_te, time.perf_counter() - t0

def run_nmf(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    min_val = X_train.min()
    X_tr = np.clip(X_train - min_val, 0, None)
    X_te = np.clip(X_test - min_val, 0, None)
    nmf = NMF(n_components=target_dim, init='random', random_state=1, max_iter=200, tol=1e-4)
    try:
        Xt = nmf.fit_transform(X_tr)
        Xs = nmf.transform(X_te)
    except ValueError:
        Xt = np.full((X_train.shape[0], target_dim), np.nan)
        Xs = np.full((X_test.shape[0], target_dim), np.nan)
    return Xt, Xs, time.perf_counter() - t0

def run_lle(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    n_n = max(target_dim + 1, min(10, X_train.shape[0] - 1))
    if X_train.shape[0] <= n_n or X_train.shape[0] <= target_dim:
        return np.full((X_train.shape[0], target_dim), np.nan), np.full((X_test.shape[0], target_dim), np.nan), 0.0
    lle = LocallyLinearEmbedding(n_components=target_dim, n_neighbors=n_n, random_state=1)
    Xt = lle.fit_transform(X_train)
    Xs = lle.transform(X_test)
    return Xt, Xs, time.perf_counter() - t0

def run_feature_agglomeration(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    n_clusters = min(max(1, target_dim), X_train.shape[1]) if X_train.shape[1] > 0 else 0
    if n_clusters > 0:
        fa = FeatureAgglomeration(n_clusters=n_clusters)
        Xt = fa.fit_transform(X_train)
        Xs = fa.transform(X_test)
    else:
        Xt = np.zeros((X_train.shape[0], 0))
        Xs = np.zeros((X_test.shape[0], 0))
    return Xt, Xs, time.perf_counter() - t0

def run_autoencoder(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    input_dim = X_train.shape[1]
    if input_dim == 0:
        return np.zeros((X_train.shape[0], target_dim)), np.zeros((X_test.shape[0], target_dim)), 0.0
    tf.keras.utils.set_random_seed(1)
    inp = Input(shape=(input_dim,))
    enc = Dense(target_dim, activation='relu')(inp)
    dec = Dense(input_dim, activation='linear')(enc)
    ae = Model(inp, dec)
    en = Model(inp, enc)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0, shuffle=True)
    Xt = en.predict(X_train, batch_size=256)
    Xs = en.predict(X_test, batch_size=256)
    K.clear_session()
    return Xt, Xs, time.perf_counter() - t0

def run_vae(X_train, X_test, target_dim):
    t0 = time.perf_counter()
    input_dim = X_train.shape[1]
    if input_dim == 0:
        return np.zeros((X_train.shape[0], target_dim)), np.zeros((X_test.shape[0], target_dim)), 0.0
    tf.keras.utils.set_random_seed(1)
    latent_dim = target_dim
    inter_dim = max(latent_dim * 2, 64)
    inp = Input(shape=(input_dim,))
    h = Dense(inter_dim, activation='relu')(inp)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    def samp(args):
        z_m, z_lv = args
        batch = K.shape(z_m)[0]
        dim = K.int_shape(z_m)[1]
        eps = K.random_normal((batch, dim))
        return z_m + K.exp(0.5 * z_lv) * eps
    z = Lambda(samp)([z_mean, z_log_var])
    dh = Dense(inter_dim, activation='relu')(z)
    out = Dense(input_dim, activation='sigmoid')(dh)
    vae = Model(inp, out)
    rec_loss = tf.keras.losses.mse(inp, out) * input_dim
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae.add_loss(K.mean(rec_loss + kl_loss))
    vae.compile(optimizer='adam')
    vae.fit(X_train, None, epochs=20, batch_size=32, verbose=0, shuffle=True)
    enc_model = Model(inp, z_mean)
    Xt = enc_model.predict(X_train, batch_size=256)
    Xs = enc_model.predict(X_test, batch_size=256)
    K.clear_session()
    return Xt, Xs, time.perf_counter() - t0

NEW_BASELINE_DR_METHODS = [
    'RandomProjection', 'FastICA', 'tSNE', 'NMF', 'LLE',
    'FeatureAgglomeration', 'Autoencoder', 'VAE'
]

# ---------------------------------------------------------
# FAISS ACCURACY HELPERS
# ---------------------------------------------------------
def get_valid_pq_m(d, max_m_val=8, ensure_d_multiple=True, min_subvector_dim=1):
    if d == 0 or d < min_subvector_dim:
        return 0
    ub = min(d // min_subvector_dim, max_m_val, d)
    for m in range(ub, 0, -1):
        if ensure_d_multiple:
            if d % m == 0:
                return m
        else:
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
        return np.array([[] for _ in range(query_data.shape[0])], dtype=int)
    k_act = min(k_neighbors, data_to_index.shape[0])
    if k_act == 0:
        return np.array([[] for _ in range(query_data.shape[0])], dtype=int)
    nbrs = NearestNeighbors(n_neighbors=k_act, algorithm='auto', n_jobs=-1).fit(data_to_index)
    inds = np.empty((len(query_data), k_act), dtype=int)
    for i in range(len(query_data)):
        inds[i] = nbrs.kneighbors(query_data[i].reshape(1, -1), return_distance=False)[0]
    return inds

def calculate_accuracy_exact_knn(gt, tr, te, k):
    t0 = time.perf_counter()
    if tr.shape[0] < 1 or te.shape[0] < 1 or gt.shape[1] == 0:
        return np.nan, 0.0
    nbrs = NearestNeighbors(n_neighbors=min(k, tr.shape[0]), n_jobs=-1).fit(tr)
    matches = sum(len(set(gt[i]) & set(nbrs.kneighbors(te[i].reshape(1, -1), return_distance=False)[0]))
                  for i in range(len(te)))
    acc = matches / (len(te) * gt.shape[1])
    return acc, time.perf_counter() - t0

def _generic_faiss_accuracy(index, gt, te, k):
    if te.shape[0] == 0 or gt.shape[1] == 0 or index is None or index.ntotal == 0:
        return 0.0
    k_act = min(k, index.ntotal)
    _, inds = index.search(te, k_act)
    matches = sum(len(set(gt[i]) & set(inds[i])) for i in range(len(te)))
    return matches / (len(te) * gt.shape[1])

def calculate_accuracy_hnswflat_faiss(gt, tr, te, k):
    t0 = time.perf_counter()
    d = tr.shape[1]
    if d == 0 or tr.shape[0] == 0:
        return np.nan, 0.0
    idx = faiss.IndexHNSWFlat(d, 32)
    idx.hnsw.efConstruction = 40
    idx.hnsw.efSearch = 50
    try:
        idx.add(tr)
    except RuntimeError as e:
        print(f"HNSWFlat add error: {e}")
        return np.nan, time.perf_counter() - t0
    acc = _generic_faiss_accuracy(idx, gt, te, k)
    return acc, time.perf_counter() - t0

def calculate_accuracy_ivfpq_faiss(gt, tr, te, k):
    t0 = time.perf_counter()
    d, n = tr.shape[1], tr.shape[0]
    if d == 0 or n == 0:
        return np.nan, 0.0
    nlist = min(100, max(1, n // 39))
    m_pq = get_valid_pq_m(d, max_m_val=8, min_subvector_dim=2)
    if m_pq == 0:
        return np.nan, 0.0
    nbits = get_dynamic_nbits(n, m_pq)
    quant = faiss.IndexFlatL2(d)
    try:
        idx = faiss.IndexIVFPQ(quant, d, nlist, m_pq, nbits)
        idx.train(tr)
        idx.add(tr)
        idx.nprobe = min(nlist, 10)
    except RuntimeError as e:
        print(f"IVFPQ error: {e}")
        return np.nan, time.perf_counter() - t0
    acc = _generic_faiss_accuracy(idx, gt, te, k)
    return acc, time.perf_counter() - t0

def calculate_accuracy_hnswpq_faiss(gt, tr, te, k):
    t0 = time.perf_counter()
    d, n = tr.shape[1], tr.shape[0]
    if d < 2 or n == 0:
        return np.nan, 0.0
    m_pq = get_valid_pq_m(d, max_m_val=8, min_subvector_dim=4) or get_valid_pq_m(d, max_m_val=8, min_subvector_dim=2)
    if m_pq == 0:
        return np.nan, 0.0
    try:
        idx = faiss.IndexHNSWPQ(d, 32, m_pq)
        idx.hnsw.efConstruction = 40
        idx.hnsw.efSearch = 50
        idx.train(tr)
        idx.add(tr)
    except RuntimeError as e:
        print(f"HNSWPQ error: {e}")
        return np.nan, time.perf_counter() - t0
    acc = _generic_faiss_accuracy(idx, gt, te, k)
    return acc, time.perf_counter() - t0

def calculate_accuracy_ivfopq_faiss(gt, tr, te, k):
    t0 = time.perf_counter()
    d, n = tr.shape[1], tr.shape[0]
    if d == 0 or n == 0:
        return np.nan, 0.0
    nlist = min(100, max(1, n // 39))
    opq_m = get_valid_pq_m(d, max_m_val=min(d,32), min_subvector_dim=2)
    if opq_m == 0:
        return np.nan, 0.0
    opq_nbits = get_dynamic_nbits(n, opq_m, min_nbits=6)
    pq_m = get_valid_pq_m(d, max_m_val=8, min_subvector_dim=2)
    if pq_m == 0:
        return np.nan, 0.0
    pq_nbits = get_dynamic_nbits(n, pq_m)
    factory = f"OPQ{opq_m}x{opq_nbits},IVF{nlist},PQ{pq_m}x{pq_nbits}"
    try:
        idx = faiss.index_factory(d, factory)
        idx.train(tr)
        idx.add(tr)
        idx.nprobe = min(nlist, 10)
    except Exception as e:
        print(f"IVFOPQ error: {e}")
        return np.nan, time.perf_counter() - t0
    acc = _generic_faiss_accuracy(idx, gt, te, k)
    return acc, time.perf_counter() - t0

# ---------------------------------------------------------
# PROCESS PARAMETERS
# ---------------------------------------------------------
def process_parameters(params_tuple, use_dw_pmad_flag, training_vectors, testing_vectors, k_values, performance_data, current_dataset_id=""):
    run_data = {'params': params_tuple, 'dataset_id': current_dataset_id}
    if use_dw_pmad_flag:
        dim_sel, ratio, b_val, alpha_val = params_tuple
    else:
        dim_sel, ratio = params_tuple
        b_val, alpha_val = "N/A", "N/A"

    actual_dim = min(dim_sel, training_vectors.shape[1])
    target_dim = max(1, int(actual_dim * ratio))
    target_dim = min(target_dim, actual_dim)
    run_data['orig_dim_selected'] = actual_dim
    run_data['target_dim_final_dr'] = target_dim

    # Standardize
    idx_sel = np.random.choice(training_vectors.shape[1], actual_dim, replace=False)
    X_train = training_vectors[:, idx_sel]
    X_test = testing_vectors[:, idx_sel]
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    run_data['time_standardization'] = time.perf_counter() - 0  # negligible
    run_data['mem_after_standardization'] = get_memory_usage()

    dr_results = {}
    dr_timings = {}
    dr_memory = {}

    # DW-PMAD (skipped if flag=False)
    if use_dw_pmad_flag:
        pass

    # Baselines
    dr_configs = {
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

    for name, (cls_or_fn, params) in dr_configs.items():
        if name == 'dw_pmad':
            continue
        start = time.perf_counter()
        n_comp = target_dim
        if name in ['pca', 'kernel_pca']:
            n_comp = min(target_dim, X_train.shape[0], X_train.shape[1])
        if name in ['umap', 'isomap', 'mds', 'tSNE', 'LLE']:
            n_comp = min(target_dim, max(1, X_train.shape[1] - 1))
        if X_train.shape[1] == 0:
            Xt = np.zeros((X_train.shape[0], 0))
            Xs = np.zeros((X_test.shape[0], 0))
            t_elapsed = 0.0
        elif name in NEW_BASELINE_DR_METHODS:
            Xt, Xs, t_elapsed = cls_or_fn(X_train, X_test, max(1, n_comp))
        else:
            p = params.copy()
            p['n_components'] = max(1, n_comp)
            model = cls_or_fn(**p)
            Xt = model.fit_transform(X_train)
            Xs = model.transform(X_test)
            t_elapsed = time.perf_counter() - start
        dr_results[name] = (Xt, Xs)
        dr_timings[name] = t_elapsed
        dr_memory[name] = get_memory_usage()

    run_data['dr_timings'] = dr_timings
    run_data['dr_memory_after'] = dr_memory

    # Accuracy
    exact_gt = {k: get_exact_neighbors(X_train, X_test, k) for k in k_values}
    acc_funcs = {
        "Exact_kNN": calculate_accuracy_exact_knn,
        "HNSWFlat_Faiss": calculate_accuracy_hnswflat_faiss,
        "IVFPQ_Faiss": calculate_accuracy_ivfpq_faiss,
        "HNSWPQ_Faiss": calculate_accuracy_hnswpq_faiss,
        "IVFOPQ_Faiss": calculate_accuracy_ivfopq_faiss,
    }
    acc_results = {k: {} for k in k_values}
    acc_times = {k: {} for k in k_values}

    for k in k_values:
        gt = exact_gt[k]
        for acc_name, func in acc_funcs.items():
            for dr_name, (tr_red, te_red) in dr_results.items():
                if "Faiss" in acc_name:
                    a, t = func(gt,
                                np.ascontiguousarray(tr_red, dtype=np.float32),
                                np.ascontiguousarray(te_red, dtype=np.float32),
                                k)
                else:
                    a, t = func(gt, tr_red, te_red, k)
                acc_results[k].setdefault(acc_name, {})[dr_name] = a
                acc_times[k].setdefault(acc_name, {})[dr_name] = t

    run_data['accuracy_results'] = acc_results
    run_data['accuracy_times'] = acc_times
    run_data['mem_after_accuracy'] = get_memory_usage()
    run_data['peak_memory_in_run'] = max(run_data['mem_after_accuracy'], run_data['mem_after_standardization'])

    performance_data.append(run_data)

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == '__main__':
    overall_start_time = time.perf_counter()
    initial_memory = get_memory_usage()

    CURRENT_DATASET_NAME = "Isolet"
    training_files_to_test = [
        f"training_vectors_{size}_{CURRENT_DATASET_NAME}.npy"
        for size in [300, 600, 900, 1200]
    ]
    test_file_global = f"testing_vectors_300_{CURRENT_DATASET_NAME}.npy"

    fixed_alpha = 6
    fixed_b = 90
    fixed_target_ratio = 0.6
    fixed_k_values = [1, 3, 6, 10, 15]
    fixed_original_dimension = 200

    # Prepare dummy files if missing
    for f_name in training_files_to_test:
        if not os.path.exists(f_name):
            num_samples = int(f_name.split('_')[2])
            dummy = np.random.rand(num_samples, fixed_original_dimension).astype(np.float32)
            np.save(f_name, dummy)
    if not os.path.exists(test_file_global):
        dummy = np.random.rand(300, fixed_original_dimension).astype(np.float32)
        np.save(test_file_global, dummy)

    # Load testing data
    if os.path.exists(test_file_global):
        testing_vectors_loaded_main_global = load_vectors(test_file_global)
    else:
        testing_vectors_loaded_main_global = np.random.rand(300, fixed_original_dimension).astype(np.float32)

    # Disable DW-PMAD in this run
    use_dw_pmad_main_execution_flag = False

    # Scheduling parallel pool
    num_parallel_workers_pool = mp.cpu_count()
    init_global_pool(num_workers=num_parallel_workers_pool)

    # Config collections
    b_values_config = [fixed_b]
    alpha_values_config = [fixed_alpha]
    k_values_for_knn_config = fixed_k_values
    dimensions_to_select_config = [fixed_original_dimension]
    target_ratios_for_dr_config = [fixed_target_ratio]

    ratio_str_for_filename = str(fixed_target_ratio).replace('.', 'p')
    params_string_for_filename = (
        f"origD{fixed_original_dimension}_"
        f"ratio{ratio_str_for_filename}_"
        f"alpha{fixed_alpha}_"
        f"b{fixed_b}"
    )

    for train_file_current in training_files_to_test:
        current_run_start = time.perf_counter()
        # Derive dataset identifier
        parts = train_file_current.split('_')
        dataset_size_id = parts[2]
        file_dataset_name_part = parts[3].split('.')[0]
        dataset_id = f"{dataset_size_id}_{CURRENT_DATASET_NAME}"

        # Load or dummy training data
        if os.path.exists(train_file_current):
            training_vectors_loaded_main = load_vectors(train_file_current)
        else:
            num_samples = int(dataset_size_id)
            training_vectors_loaded_main = np.random.rand(num_samples, fixed_original_dimension).astype(np.float32)

        # Parameter combinations (only dim & ratio, since DW-PMAD is disabled)
        parameter_combinations_for_run_list = list(itertools.product(
            dimensions_to_select_config,
            target_ratios_for_dr_config
        ))

        current_training_file_performance_data_list = []
        for current_params_tuple in tqdm(parameter_combinations_for_run_list,
                                         desc=f"Processing {train_file_current}"):
            process_parameters(
                current_params_tuple,
                use_dw_pmad_main_execution_flag,
                training_vectors_loaded_main,
                testing_vectors_loaded_main_global,
                k_values_for_knn_config,
                current_training_file_performance_data_list,
                current_dataset_id=dataset_id
            )

        # --- Reporting Section (Excel/CSV export) ---
        print(f"\nProcessing results for dataset: {dataset_id} (Params: {params_string_for_filename})")
        excel_writer_data_frames = {}
        ann_sheets = ["Exact_kNN", "HNSWFlat_Faiss", "IVFPQ_Faiss", "HNSWPQ_Faiss", "IVFOPQ_Faiss"]
        for sheet in ann_sheets:
            excel_writer_data_frames[sheet] = []

        dr_order = [
            'dw_pmad', 'pca', 'umap', 'isomap', 'kernel_pca', 'mds',
            'RandomProjection', 'FastICA', 'tSNE', 'NMF', 'LLE',
            'FeatureAgglomeration', 'Autoencoder', 'VAE'
        ]

        for run_item in current_training_file_performance_data_list:
            base = {
                'Dataset_ID': run_item['dataset_id'],
                'Dimension_Selected_Config': run_item['params'][0],
                'Target_Ratio_DR_Config': run_item['params'][1],
            }
            if len(run_item['params']) > 3:
                base['b_dwpmad_Config'] = run_item['params'][2]
                base['alpha_dwpmad_Config'] = run_item['params'][3]
            else:
                base['b_dwpmad_Config'] = "N/A"
                base['alpha_dwpmad_Config'] = "N/A"
            base['Orig_Dim_Actual_Selected'] = run_item['orig_dim_selected']
            base['Target_Dim_Actual_DR'] = run_item['target_dim_final_dr']

            for k_val in k_values_for_knn_config:
                row_base = base.copy()
                row_base['k_Neighbors'] = k_val
                # DR times
                for dr in dr_order:
                    row_base[f'{dr}_DR_Time'] = run_item['dr_timings'].get(dr, np.nan)
                # Accuracy & times
                for sheet in ann_sheets:
                    row = row_base.copy()
                    for dr in dr_order:
                        row[f'{dr}_Accuracy'] = np.nan
                        row[f'{dr}_Accuracy_Time'] = np.nan
                    accs = run_item['accuracy_results'].get(k_val, {}).get(sheet, {})
                    times = run_item['accuracy_times'].get(k_val, {}).get(sheet, {})
                    for dr, acc_val in accs.items():
                        row[f'{dr}_Accuracy'] = acc_val
                        row[f'{dr}_Accuracy_Time'] = times.get(dr, np.nan)
                    excel_writer_data_frames[sheet].append(row)

        output_excel = f'makeup_scalability_ANN_SOTA_results_{dataset_id}_{params_string_for_filename}.xlsx'
        output_csv_prefix = f'makeup_scalability_ANN_SOTA_results_{dataset_id}_{params_string_for_filename}'

        cols_first = [
            'Dataset_ID', 'Dimension_Selected_Config', 'Target_Ratio_DR_Config',
            'b_dwpmad_Config', 'alpha_dwpmad_Config',
            'Orig_Dim_Actual_Selected', 'Target_Dim_Actual_DR', 'k_Neighbors'
        ]
        ordered_cols = []
        for dr in dr_order:
            ordered_cols += [f'{dr}_Accuracy', f'{dr}_DR_Time', f'{dr}_Accuracy_Time']
        final_cols = cols_first + ordered_cols

        try:
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                for sheet, data in excel_writer_data_frames.items():
                    if data:
                        df = pd.DataFrame(data)
                        for col in final_cols:
                            if col not in df.columns:
                                df[col] = np.nan
                        df = df[final_cols]
                        df.to_excel(writer, sheet_name=sheet, index=False)
                    else:
                        print(f"No data for sheet: {sheet}")
            print(f"Exported Excel: {output_excel}")
        except ImportError:
            for sheet, data in excel_writer_data_frames.items():
                if data:
                    df = pd.DataFrame(data)
                    for col in final_cols:
                        if col not in df.columns:
                            df[col] = np.nan
                    df = df[final_cols]
                    csv_file = f"{output_csv_prefix}_{sheet}.csv"
                    df.to_csv(csv_file, index=False)
                    print(f"Exported CSV: {csv_file}")

        # --- Performance report text file ---
        report_file = f'makeup_performance_report_SOTA_ANN_{dataset_id}_{params_string_for_filename}.txt'
        agg_dr_times = {dr: 0.0 for dr in dr_order}
        agg_acc_times = {sheet: {dr: 0.0 for dr in dr_order} for sheet in ann_sheets}

        for item in current_training_file_performance_data_list:
            for dr, tval in item['dr_timings'].items():
                if dr in agg_dr_times:
                    agg_dr_times[dr] += tval if not np.isnan(tval) else 0.0
            for k_val, sheets in item['accuracy_times'].items():
                for sheet, dr_times in sheets.items():
                    for dr, tv in dr_times.items():
                        if sheet in agg_acc_times and dr in agg_acc_times[sheet]:
                            agg_acc_times[sheet][dr] += tv if not np.isnan(tv) else 0.0

        with open(report_file, 'w') as f:
            f.write(f"SOTA ANN Performance Report ({CURRENT_DATASET_NAME}, Dataset {dataset_id}, Params {params_string_for_filename})\n\n")
            f.write(f"Total Iteration Time: {time.perf_counter() - current_run_start:.4f}s\n")
            f.write(f"Initial Memory: {initial_memory / (1024**2):.2f} MB\n")
            f.write(f"Memory after loading: {get_memory_usage() / (1024**2):.2f} MB\n\n")
            f.write("Aggregated DR Timings:\n")
            for dr, ttot in agg_dr_times.items():
                f.write(f"  - {dr}: {ttot:.4f}s\n")
            f.write("\nAggregated Accuracy Timings:\n")
            for sheet, dr_times in agg_acc_times.items():
                f.write(f"  {sheet}:\n")
                for dr, ttot in dr_times.items():
                    f.write(f"    - {dr}: {ttot:.4f}s\n")
        print(f"Generated report: {report_file}")

    # Close pool
    if global_pool:
        global_pool.close()
        global_pool.join()
        print("Global multiprocessing pool closed.")

    total_time = time.perf_counter() - overall_start_time
    print(f"\nTotal script execution time: {total_time:.4f}s")
