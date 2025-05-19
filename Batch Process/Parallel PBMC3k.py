import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
import time
import os # For performance report
import psutil # For memory usage
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import pdist # Kept for comparison if needed, but parallel_pdist is used
from tqdm import tqdm
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
import umap # Standard import; ensure this is from umap-learn
import faiss # For ANN methods

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
        partitioned_idx = max(0, top_b_count -1)
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
            print(f"Warning: DW-PMAD optimization for axis {axis_num+1} did not converge optimally (Message: {result.message}). Using resulting vector.")
            optimal_w_found = result.x / (np.linalg.norm(result.x) + 1e-9)

        prev_ws_list.append(optimal_w_found.astype(np.float32))
        optimal_ws_list.append(optimal_w_found.astype(np.float32))

    projection_axes_found = np.column_stack(optimal_ws_list)
    projected_X_result = X @ projection_axes_found
    total_time_taken = time.perf_counter() - total_start_time
    print(f"DW-PMAD timing: {total_time_taken:.4f}s")
    return projected_X_result, projection_axes_found


def project_dw_pmad(X_to_project, projection_axes_to_use):
    return X_to_project @ projection_axes_to_use

# ---------------------------------------------------------
# FAISS HELPER
# ---------------------------------------------------------
def get_valid_pq_m(d, max_m_val=8, ensure_d_multiple=True, min_subvector_dim=1):
    if d == 0: return 0
    if d < min_subvector_dim : return 0

    upper_bound_m = min(d // min_subvector_dim if min_subvector_dim > 0 else d, max_m_val, d)
    if upper_bound_m == 0 and d > 0 : # If d is small, min_subvector_dim might make upper_bound_m zero
        upper_bound_m = 1 # M must be at least 1 if d > 0

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
    # Heuristic: if n_train is significantly less than M * (factor * 2^nbits), reduce nbits
    # Factor of 4: if n_train < M * 64 (for 8 bits, 256 centroids)
    # This aims to have at least (factor) points per centroid on average for each sub-quantizer's codebook.
    # Since PQ trains M codebooks, and n_train is available for each, we consider 2^nbits.
    # If n_train < factor_per_centroid * 2^nbits, then it's low.
    # Let factor_per_centroid be e.g. 4. So if n_train < 4 * 2^nbits.
    # This check is per sub-quantizer, so m_pq is not directly in this heuristic's threshold.

    # If n_train is less than, say, 4 times the number of centroids (2^nbits)
    while n_train < 4 * (2**nbits) and nbits > min_nbits:
        nbits -= 1

    # print(f"  Dynamically adjusting PQ nbits to {nbits} due to low n_train ({n_train}) for m_pq={m_pq}.")
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
    # print(f"  IVFPQ using m_pq={m_pq}, nbits_pq={nbits_pq} for d={dim_reduced}, n_train={n_train}")

    quantizer = faiss.IndexFlatL2(dim_reduced)
    index = None
    try:
        index = faiss.IndexIVFPQ(quantizer, dim_reduced, nlist, m_pq, nbits_pq)
        if n_train < 39 * nlist :
            print(f"Faiss IVFPQ Warning: n_train ({n_train}) < Faiss heuristic min training points ({39*nlist} for nlist={nlist}). Quality may be affected.")

        if n_train < (2**nbits_pq) * 4 : # Heuristic: need at least ~4 points per PQ codebook entry
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

    # HNSWPQ's internal PQ uses 8 bits. M must divide d.
    # d/M should ideally be >= 2 (or even >=4 for some PQ implementations).
    m_pq_to_try = get_valid_pq_m(dim_reduced, max_m_val=8, min_subvector_dim=4)

    if m_pq_to_try == 0:
        print(f"HNSWPQ Skipping: Could not find valid m_pq for d={dim_reduced} with min_subvector_dim=2 (required for HNSWPQ's internal PQ).")
        return np.nan, 0.0

    index = None
    try:
        # print(f"  HNSWPQ Attempt: d={dim_reduced}, m_pq={m_pq_to_try}")
        index = faiss.IndexHNSWPQ(dim_reduced, M_hnsw, m_pq_to_try)
        index.hnsw.efConstruction = efConstruction_hnsw
        index.hnsw.efSearch = efSearch_hnsw

        # PQ training heuristic for HNSWPQ's internal PQ (which uses 8 bits)
        if n_train < m_pq_to_try * (2**8 / 8) : # e.g. < m_pq * 32 points
            print(f"Faiss HNSWPQ Warning: n_train ({n_train}) might be too small for PQ training (m_pq={m_pq_to_try}, internal nbits=8).")

        index.train(reduced_data_train_faiss)
        index.add(reduced_data_train_faiss)
    except RuntimeError as e:
        if "d % M == 0" in str(e):
            print(f"HNSWPQ specific 'd % M' RuntimeError (d={dim_reduced}, m_pq={m_pq_to_try}): {e}. This index type appears unstable for this d/m_pq. Skipping HNSWPQ.")
        elif "sub-vector size" in str(e) or "pq.dsub" in str(e): # Catch other potential PQ related errors for HNSWPQ
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

    # M for OPQ's internal PQ training.
    opq_train_m = get_valid_pq_m(dim_reduced, max_m_val=min(dim_reduced, 32), ensure_d_multiple=True, min_subvector_dim=2)
    if opq_train_m == 0:
        print(f"IVFOPQ Skipping: Could not find valid opq_train_m for d={dim_reduced} with min_subvector_dim=2.")
        return np.nan, 0.0
    opq_nbits = get_dynamic_nbits(n_train, opq_train_m, default_nbits=8, min_nbits=6) # Allow 6 bits for OPQ's PQ

    # M for the final PQ stage.
    final_pq_m = get_valid_pq_m(dim_reduced, max_m_val=8, ensure_d_multiple=True, min_subvector_dim=2)
    if final_pq_m == 0:
        print(f"IVFOPQ Skipping: Could not find valid final_pq_m for d={dim_reduced} with min_subvector_dim=2.")
        return np.nan, 0.0
    final_nbits = get_dynamic_nbits(n_train, final_pq_m, default_nbits=8, min_nbits=4)

    # Factory string: OPQ pre-transform, then IVF, then PQ on residuals.
    # OPQ<M_opq>[x<nbits_opq_pq>],IVF<nlist>,PQ<M_final_pq>[x<nbits_final_pq>]
    factory_str = f"OPQ{opq_train_m}x{opq_nbits},IVF{nlist},PQ{final_pq_m}x{final_nbits}"
    # print(f"  IVFOPQ using factory: '{factory_str}' for d={dim_reduced}, n_train={n_train}")

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
def process_parameters(params_tuple, use_dw_pmad_flag_param, training_vectors_full_param, testing_vectors_full_param, k_values_global_param, performance_data_collector_list):
    run_performance_data_dict = {'params': params_tuple}
    current_peak_memory_usage = get_memory_usage()

    if use_dw_pmad_flag_param:
        dim_from_params, target_ratio_from_params, b_from_params, alpha_from_params = params_tuple
    else:
        dim_from_params, target_ratio_from_params = params_tuple
        b_from_params, alpha_from_params = "N/A", "N/A"

    actual_dim_selected_from_orig = min(dim_from_params, training_vectors_full_param.shape[1])
    target_dim_for_dr_final = max(1, int(actual_dim_selected_from_orig * target_ratio_from_params))
    target_dim_for_dr_final = min(target_dim_for_dr_final, actual_dim_selected_from_orig)

    print(f"\nProcessing: Params={params_tuple}, Orig Dim Selected={actual_dim_selected_from_orig}, Final DR Target Dim={target_dim_for_dr_final}")
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
    # print(f"Data standardization complete in {run_performance_data_dict['time_standardization']:.4f}s")

    dr_methods_results_output_dict = {}
    dr_timings_output_dict = {}
    dr_memory_after_output_dict = {}

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

    dr_method_configs = {
        'pca': (PCA, {'random_state': 1}),
        'umap': (umap.UMAP, {'random_state': 1, 'min_dist': 0.1, 'n_jobs': 1}),
        'isomap': (Isomap, {}),
        'kernel_pca': (KernelPCA, {'kernel': 'rbf', 'random_state': 1, 'n_jobs': -1}),
        'mds': (MDS, {'dissimilarity': 'euclidean', 'random_state': 1, 'normalized_stress': 'auto', 'n_jobs': -1})
    }

    for dr_method_name_key, (model_class_val, model_params_dict) in dr_method_configs.items():
        print(f"Starting {dr_method_name_key.upper()}...")
        dr_method_start_time = time.perf_counter()

        current_n_components_for_dr = target_dim_for_dr_final
        if dr_method_name_key == 'pca':
            current_n_components_for_dr = min(target_dim_for_dr_final, X_train_standardized_data.shape[0], X_train_standardized_data.shape[1])
        elif dr_method_name_key in ['umap', 'isomap', 'mds']:
            if X_train_standardized_data.shape[1] > 1:
                current_n_components_for_dr = min(target_dim_for_dr_final, X_train_standardized_data.shape[1] -1)
            else:
                current_n_components_for_dr = min(target_dim_for_dr_final, 1)

        current_n_components_for_dr = max(1, current_n_components_for_dr) if X_train_standardized_data.shape[1] > 0 else 0
        model_params_dict['n_components'] = current_n_components_for_dr

        skip_dr = False
        X_reduced_train_val, X_reduced_test_val = None, None

        if X_train_standardized_data.shape[1] == 0:
            skip_dr = True
            X_reduced_train_val = np.zeros((X_train_standardized_data.shape[0], 0), dtype=X_train_standardized_data.dtype)
            X_reduced_test_val = np.zeros((X_test_standardized_data.shape[0], 0), dtype=X_test_standardized_data.dtype)
        elif current_n_components_for_dr == 0 :
            skip_dr = True
            X_reduced_train_val = np.zeros((X_train_standardized_data.shape[0], 0), dtype=X_train_standardized_data.dtype)
            X_reduced_test_val = np.zeros((X_test_standardized_data.shape[0], 0), dtype=X_test_standardized_data.dtype)
        elif X_train_standardized_data.shape[0] <= current_n_components_for_dr and dr_method_name_key in ['umap', 'isomap', 'mds']:
            skip_dr = True
        elif dr_method_name_key == 'mds' and X_train_standardized_data.shape[0] > 3000:
            skip_dr = True

        if skip_dr and X_reduced_train_val is None:
            target_dim_for_fallback_skip = current_n_components_for_dr if current_n_components_for_dr > 0 else 1
            X_reduced_train_val = np.full((X_train_standardized_data.shape[0], target_dim_for_fallback_skip), np.nan)
            X_reduced_test_val = np.full((X_test_standardized_data.shape[0], target_dim_for_fallback_skip), np.nan)
        elif not skip_dr :
            try:
                if dr_method_name_key in ['umap', 'isomap']:
                    n_neigh = min(15, X_train_standardized_data.shape[0] - 1 if X_train_standardized_data.shape[0] > 1 else 1)
                    model_params_dict['n_neighbors'] = max(1, n_neigh)

                model_instance = model_class_val(**model_params_dict)
                X_reduced_train_val = model_instance.fit_transform(X_train_standardized_data)

                if dr_method_name_key == 'mds':
                    if X_reduced_train_val.ndim == 1: X_reduced_train_val = X_reduced_train_val.reshape(-1,1)
                    if X_train_standardized_data.shape[0] > 0 and X_reduced_train_val.shape[0] > 0 :
                        regressor_mds = LinearRegression().fit(X_train_standardized_data, X_reduced_train_val)
                        X_reduced_test_val = regressor_mds.predict(X_test_standardized_data)
                    else:
                        X_reduced_test_val = np.full((X_test_standardized_data.shape[0], X_reduced_train_val.shape[1] if X_reduced_train_val.ndim >1 and X_reduced_train_val.shape[1] > 0 else 1), np.nan)
                else:
                    X_reduced_test_val = model_instance.transform(X_test_standardized_data)
            except Exception as e_dr:
                print(f"{dr_method_name_key.upper()} error: {e_dr}. Filling with NaNs.")
                target_dim_for_fallback_error = model_params_dict.get('n_components', 1)
                X_reduced_train_val = np.full((X_train_standardized_data.shape[0], target_dim_for_fallback_error), np.nan)
                X_reduced_test_val = np.full((X_test_standardized_data.shape[0], target_dim_for_fallback_error), np.nan)

        dr_methods_results_output_dict[dr_method_name_key] = (X_reduced_train_val, X_reduced_test_val)
        dr_timings_output_dict[dr_method_name_key] = time.perf_counter() - dr_method_start_time
        dr_memory_after_output_dict[dr_method_name_key] = get_memory_usage()
        current_peak_memory_usage = max(current_peak_memory_usage, dr_memory_after_output_dict[dr_method_name_key])
        output_shape_str = X_reduced_train_val.shape if X_reduced_train_val is not None else "DR_Failed/Skipped"
        print(f"{dr_method_name_key.upper()} complete in {dr_timings_output_dict[dr_method_name_key]:.4f}s. Output shape: {output_shape_str}")

    run_performance_data_dict['dr_timings'] = dr_timings_output_dict
    run_performance_data_dict['dr_memory_after'] = dr_memory_after_output_dict

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
                    pd.isna(X_reduced_train_iter).all() or pd.isna(X_reduced_test_iter).all() or \
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

    b_values_config = [60,70,80,90, 100]
    alpha_values_config = [1, 6, 12, 18, 25, 35, 50, 10000]
    k_values_for_knn_config = [1, 3, 6, 10, 15]
    dimensions_to_select_config = [200]
    target_ratios_for_dr_config = [0.05, 0.1, 0.2, 0.4, 0.6]


    print("Loading data...")
    try:
        train_file = 'training_vectors_1200_PBMC3k.npy'
        test_file = 'testing_vectors_300_PBMC3k.npy'

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"Warning: Dataset files ('{train_file}', '{test_file}') not found.")
            print("Creating dummy data for demonstration.")

            dummy_feature_size_val = max(dimensions_to_select_config) if dimensions_to_select_config else 200
            dummy_feature_size_val = max(dummy_feature_size_val, 50)

            training_vectors_loaded_main = np.random.rand(600, dummy_feature_size_val).astype(np.float32)
            testing_vectors_loaded_main = np.random.rand(300, dummy_feature_size_val).astype(np.float32)
            print(f"Using dummy training_vectors shape: {training_vectors_loaded_main.shape}")
            print(f"Using dummy testing_vectors shape: {testing_vectors_loaded_main.shape}")
        else:
            training_vectors_loaded_main = load_vectors(train_file)
            testing_vectors_loaded_main = load_vectors(test_file)
            print(f"Loaded actual training_vectors shape: {training_vectors_loaded_main.shape}")
            print(f"Loaded actual testing_vectors shape: {testing_vectors_loaded_main.shape}")

    except FileNotFoundError as e_fnf:
        print(f"Error: Dataset file not found - {e_fnf}.")
        exit()
    except Exception as e_load:
        print(f"An error occurred during data loading: {e_load}.")
        exit()

    mem_after_data_load_main = get_memory_usage()
    print(f"Data loading/creation complete. Memory usage: {mem_after_data_load_main / (1024**2):.2f} MB")

    print("\n--- Experiment Parameter Settings ---")
    print(f"DW-PMAD 'b' values: {b_values_config}")
    print(f"DW-PMAD 'alpha' values: {alpha_values_config}")
    print(f"k-NN 'k' values for accuracy: {k_values_for_knn_config}")
    print(f"Initial dimensions to select from dataset: {dimensions_to_select_config}")
    print(f"Target DR ratios (of selected dimension): {target_ratios_for_dr_config}")

    use_dw_pmad_input_str = 'y'
    use_dw_pmad_main_execution_flag = not use_dw_pmad_input_str.startswith('n')

    if use_dw_pmad_main_execution_flag:
        parameter_combinations_for_run_list = list(itertools.product(
            dimensions_to_select_config,
            target_ratios_for_dr_config,
            b_values_config,
            alpha_values_config
        ))
    else:
        parameter_combinations_for_run_list = list(itertools.product(
            dimensions_to_select_config,
            target_ratios_for_dr_config
        ))

    num_parallel_workers_pool = mp.cpu_count()
    init_global_pool(num_workers=num_parallel_workers_pool)

    all_run_performance_data_collected_list = []
    for current_params_tuple_iter in tqdm(parameter_combinations_for_run_list, desc="Processing Parameter Combinations"):
        process_parameters(
            current_params_tuple_iter,
            use_dw_pmad_main_execution_flag,
            training_vectors_loaded_main,
            testing_vectors_loaded_main,
            k_values_for_knn_config,
            all_run_performance_data_collected_list
        )

    if global_pool:
        global_pool.close()
        global_pool.join()
        print("Global multiprocessing pool closed.")

    print("\nProcessing results for Excel/CSV export...")
    excel_writer_data_frames_dict = {}
    ann_method_names_for_excel_sheets = ["Exact_kNN", "HNSWFlat_Faiss", "IVFPQ_Faiss", "HNSWPQ_Faiss", "IVFOPQ_Faiss"]

    for acc_method_sheet_name in ann_method_names_for_excel_sheets:
        excel_writer_data_frames_dict[acc_method_sheet_name] = []

    dr_method_names_ordered_for_excel = ['dw_pmad', 'pca', 'umap', 'isomap', 'kernel_pca', 'mds']
    if not use_dw_pmad_main_execution_flag:
        if 'dw_pmad' in dr_method_names_ordered_for_excel:
            dr_method_names_ordered_for_excel.remove('dw_pmad')

    for run_data_item_excel in all_run_performance_data_collected_list:
        base_info_for_row_dict = {}
        if use_dw_pmad_main_execution_flag:
            base_info_for_row_dict['Dimension_Selected_Config'] = run_data_item_excel['params'][0]
            base_info_for_row_dict['Target_Ratio_DR_Config'] = run_data_item_excel['params'][1]
            base_info_for_row_dict['b_dwpmad_Config'] = run_data_item_excel['params'][2]
            base_info_for_row_dict['alpha_dwpmad_Config'] = run_data_item_excel['params'][3]
        else:
            base_info_for_row_dict['Dimension_Selected_Config'] = run_data_item_excel['params'][0]
            base_info_for_row_dict['Target_Ratio_DR_Config'] = run_data_item_excel['params'][1]
            base_info_for_row_dict['b_dwpmad_Config'] = "N/A"
            base_info_for_row_dict['alpha_dwpmad_Config'] = "N/A"
        base_info_for_row_dict['Orig_Dim_Actual_Selected'] = run_data_item_excel['orig_dim_selected']
        base_info_for_row_dict['Target_Dim_Actual_DR'] = run_data_item_excel['target_dim_final_dr']

        for k_val_iter_excel in k_values_for_knn_config:
            row_for_k_val_excel = base_info_for_row_dict.copy()
            row_for_k_val_excel['k_Neighbors'] = k_val_iter_excel

            for acc_method_sheet_name_iter in ann_method_names_for_excel_sheets:
                current_row_for_sheet = row_for_k_val_excel.copy()
                for dr_method_col_name in dr_method_names_ordered_for_excel:
                    current_row_for_sheet[f'{dr_method_col_name}_Accuracy'] = np.nan

                if k_val_iter_excel in run_data_item_excel['accuracy_results'] and \
                        acc_method_sheet_name_iter in run_data_item_excel['accuracy_results'][k_val_iter_excel]:
                    for dr_method_name_found, acc_val_found_excel in run_data_item_excel['accuracy_results'][k_val_iter_excel][acc_method_sheet_name_iter].items():
                        if dr_method_name_found in dr_method_names_ordered_for_excel:
                            current_row_for_sheet[f'{dr_method_name_found}_Accuracy'] = acc_val_found_excel

                excel_writer_data_frames_dict[acc_method_sheet_name_iter].append(current_row_for_sheet)

    output_excel_filename_val = 'parameter_sweep_ANN_SOTA_results_PBMC3k.xlsx'
    output_csv_prefix = 'parameter_sweep_ANN_SOTA_results_PBMC3k'

    try:
        with pd.ExcelWriter(output_excel_filename_val, engine='openpyxl') as writer_obj_excel:
            for acc_method_sheet_name_write, data_list_for_sheet_write in excel_writer_data_frames_dict.items():
                if data_list_for_sheet_write:
                    df_for_sheet_write = pd.DataFrame(data_list_for_sheet_write)
                    cols_first_part_excel = ['Dimension_Selected_Config', 'Target_Ratio_DR_Config',
                                             'b_dwpmad_Config', 'alpha_dwpmad_Config',
                                             'Orig_Dim_Actual_Selected', 'Target_Dim_Actual_DR', 'k_Neighbors']

                    ordered_accuracy_cols = [f'{dr_name}_Accuracy' for dr_name in dr_method_names_ordered_for_excel]
                    final_cols_order_excel = cols_first_part_excel + ordered_accuracy_cols

                    for col_ensure_excel in final_cols_order_excel:
                        if col_ensure_excel not in df_for_sheet_write.columns:
                            df_for_sheet_write[col_ensure_excel] = np.nan

                    df_for_sheet_write = df_for_sheet_write[final_cols_order_excel]
                    df_for_sheet_write.to_excel(writer_obj_excel, sheet_name=acc_method_sheet_name_write, index=False)
                else:
                    print(f"No data to write for Excel sheet: {acc_method_sheet_name_write}")
        print(f"Results exported to Excel: '{output_excel_filename_val}'")
    except ImportError:
        print("Module 'openpyxl' not found. Falling back to saving results as individual CSV files.")
        for acc_method_sheet_name_csv, data_list_for_csv_write in excel_writer_data_frames_dict.items():
            if data_list_for_csv_write:
                df_for_csv_write = pd.DataFrame(data_list_for_csv_write)
                cols_first_part_csv = ['Dimension_Selected_Config', 'Target_Ratio_DR_Config',
                                       'b_dwpmad_Config', 'alpha_dwpmad_Config',
                                       'Orig_Dim_Actual_Selected', 'Target_Dim_Actual_DR', 'k_Neighbors']
                ordered_accuracy_cols_csv = [f'{dr_name}_Accuracy' for dr_name in dr_method_names_ordered_for_excel]
                final_cols_order_csv = cols_first_part_csv + ordered_accuracy_cols_csv
                for col_ensure_csv in final_cols_order_csv:
                    if col_ensure_csv not in df_for_csv_write.columns:
                        df_for_csv_write[col_ensure_csv] = np.nan
                df_for_csv_write = df_for_csv_write[final_cols_order_csv]

                csv_filename = f"{output_csv_prefix}_{acc_method_sheet_name_csv}.csv"
                df_for_csv_write.to_csv(csv_filename, index=False)
                print(f"Results for {acc_method_sheet_name_csv} exported to CSV: '{csv_filename}'")
            else:
                print(f"No data to write for CSV: {acc_method_sheet_name_csv}")


    print("Generating performance report...")
    overall_end_time_script_exec_final = time.perf_counter()
    total_execution_time_for_script = overall_end_time_script_exec_final - overall_start_time_script_exec

    aggregated_dr_method_times_report = {dr_name: 0.0 for dr_name in dr_method_names_ordered_for_excel}
    aggregated_accuracy_method_times_report = {
        acc_m_name: {dr_name: 0.0 for dr_name in dr_method_names_ordered_for_excel}
        for acc_m_name in ann_method_names_for_excel_sheets
    }

    for run_data_item_perf_report in all_run_performance_data_collected_list:
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

    output_report_filename_val = 'performance_report_SOTA_ANN_PBMC3k.txt'
    with open(output_report_filename_val, 'w') as f_report_obj:
        f_report_obj.write("--- SOTA ANN Performance Report (PBMC3k Dataset) ---\n\n")
        f_report_obj.write(f"Total Script Execution Time: {total_execution_time_for_script:.4f}s\n")
        f_report_obj.write(f"Initial Memory Usage (at script start): {initial_memory_at_script_start / (1024**2):.2f} MB\n")
        f_report_obj.write(f"Memory Usage (after data load/creation): {mem_after_data_load_main / (1024**2):.2f} MB\n\n")

        f_report_obj.write("--- Aggregated Timings ---\n")
        f_report_obj.write("Dimensionality Reduction Methods (Sum Total Time over all runs):\n")
        for dr_method_name_agg_report, total_t_agg_report in aggregated_dr_method_times_report.items():
            f_report_obj.write(f"  - {dr_method_name_agg_report}: {total_t_agg_report:.4f}s\n")

        f_report_obj.write("\nAccuracy Checking Methods (Sum Total Time, Summed over all k-values and runs, per DR method):\n")
        for acc_method_name_agg_report, dr_times_data_agg_report in aggregated_accuracy_method_times_report.items():
            f_report_obj.write(f"  - Accuracy Method: {acc_method_name_agg_report}\n")
            for dr_method_name_agg_inner_report, total_t_acc_agg_report in dr_times_data_agg_report.items():
                f_report_obj.write(f"    - On {dr_method_name_agg_inner_report} reduced data: {total_t_acc_agg_report:.4f}s\n")

        f_report_obj.write("\n--- Detailed Timings and Memory per Parameter Combination Run ---\n")
        for i, run_data_item_detail_report in enumerate(all_run_performance_data_collected_list):
            f_report_obj.write(f"\nRun {i+1}: Params Config: {run_data_item_detail_report['params']}, "
                               f"Actual Orig Dim: {run_data_item_detail_report['orig_dim_selected']}, "
                               f"Final DR Target Dim: {run_data_item_detail_report['target_dim_final_dr']}\n")
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

    print(f"Performance report generated: '{output_report_filename_val}'")
    print(f"Total script execution time: {total_execution_time_for_script:.4f}s")

