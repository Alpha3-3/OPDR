#!/usr/bin/env python3
"""
Main program variant: keep 11 DR methods serial, but run MPAD with CPU multi-core parallelism.

How parallelism is achieved inside MPAD in this program:
- We maximize CPU core usage for all NumPy/BLAS-heavy ops used by MPAD by setting
  OMP/MKL/OPENBLAS thread counts to the machine's CPU core count at runtime.
- Where available, we also use threadpoolctl to set thread pools for NumPy/BLAS
  libraries (MKL/OpenBLAS/BLIS) to the same value. This makes the large matrix
  ops inside MPAD (projection/gradient steps) run multi-threaded without changing
  MPAD's optimization logic.

NOTE: We do NOT modify the original MPAD logic. All 11 DR methods still execute
serially as requested. Only MPAD's underlying numerical kernels are configured
to run multi-threaded on CPU.
"""

import os
import multiprocessing as mp

# Prefer explicit thread control if available
try:
    from threadpoolctl import threadpool_limits
except Exception:  # threadpoolctl is optional
    threadpool_limits = None  # type: ignore

import numpy as np

# Reuse existing components from the main program
from main_program import (
    MPAD,                      # original MPAD implementation
    BaselineMethods,           # all baseline DR methods
    IndexMethods,              # ANN indices
    evaluate_method,           # detailed timing + memory collection
    save_results_to_csv,
    save_summary_report,
)


def _set_max_cpu_threads(max_threads: int) -> None:
    """Force BLAS backends to use up to max_threads on CPU.

    This affects NumPy/Scipy linear algebra used inside MPAD. On Linux this will
    typically control MKL/OpenBLAS/BLIS; on Windows it will control MKL when present.
    """
    max_threads = max(1, int(max_threads))
    # Environment hints (picked up by most BLAS backends)
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)


def run_mpad_parallel(x_train: np.ndarray, x_test: np.ndarray, target_dim: int,
                      b_percentage: float = 1.0, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Run MPAD with CPU multi-core parallelism enabled for BLAS ops.

    - Sets BLAS backends to use all available CPU cores for the duration of the call.
    - Constructs and runs the original MPAD, unchanged.
    """
    max_threads = mp.cpu_count()

    if threadpool_limits is not None:
        # Use context manager to set threadpools for the duration of this call
        with threadpool_limits(limits=max_threads):
            _set_max_cpu_threads(max_threads)
            mpad = MPAD(b_percentage=b_percentage, alpha=alpha, target_dim=target_dim)
            x_train_red = mpad.fit_transform(x_train)
            x_test_red = mpad.transform(x_test)
            return x_train_red, x_test_red
    else:
        # Fallback to environment-based control
        _set_max_cpu_threads(max_threads)
        mpad = MPAD(b_percentage=b_percentage, alpha=alpha, target_dim=target_dim)
        x_train_red = mpad.fit_transform(x_train)
        x_test_red = mpad.transform(x_test)
        return x_train_red, x_test_red


def main_evaluation_mpad_parallel(dataset_name: str,
                                  train_file: str,
                                  test_file: str,
                                  target_dim: int,
                                  b_percentage: float,
                                  alpha: float,
                                  k_values: list[int],
                                  save_results: bool = True,
                                  output_dir: str = "Result"):
    """Main evaluation variant that runs MPAD with CPU multi-threaded BLAS.

    All 11 DR methods remain serial. Only MPAD's numerical kernels will use
    multi-core CPU via BLAS thread control.
    """
    # Load data
    X_train = np.load(train_file)
    X_test = np.load(test_file)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Methods: replace MPAD entry with the parallelized runner, keep others unchanged
    methods = {
        'MPAD': lambda X_tr, X_te, td: run_mpad_parallel(X_tr, X_te, td, b_percentage=b_percentage, alpha=alpha),
        'PCA': BaselineMethods.run_pca,
        'UMAP': BaselineMethods.run_umap,
        'Isomap': BaselineMethods.run_isomap,
        'KernelPCA': BaselineMethods.run_kernel_pca,
        'RandomProjection': BaselineMethods.run_random_projection,
        # 'tSNE': BaselineMethods.run_tsne,  # intentionally disabled
        'NMF': BaselineMethods.run_nmf,
        'LLE': BaselineMethods.run_lle,
        'FeatureAgglomeration': BaselineMethods.run_feature_agglomeration,
        'Autoencoder': BaselineMethods.run_autoencoder,
        'VAE': BaselineMethods.run_vae,
    }

    index_methods = {
        'IndexFlat_kNN': IndexMethods.exact_knn,
        'HNSWFlat': IndexMethods.hnswflat_faiss,
        'IVFPQ': IndexMethods.ivfpq_faiss,
        'IVF_PQR': IndexMethods.ivf_pqr_faiss,
        'IVF_OPQ_PQ': IndexMethods.ivf_opq_pq_faiss,
    }

    # Evaluate all methods serially (unchanged behavior)
    all_results: dict[str, dict] = {}
    total_methods = len(methods)

    print(f"\n[PROGRESS] Starting evaluation of {total_methods} dimensionality reduction methods (MPAD uses CPU multi-core)")
    print(f"[PROGRESS] Each method will be tested with {len(index_methods)-1} index methods and {len(k_values)} k values")

    for i, (method_name, method_func) in enumerate(methods.items(), 1):
        print(f"\n[PROGRESS] Method {i}/{total_methods}: {method_name}")
        print("=" * 60)

        results = evaluate_method(method_name, method_func, X_train, X_test, target_dim, index_methods, k_values)
        all_results[method_name] = results

    print(f"\n[PROGRESS] All {total_methods} methods completed!")

    # Save results if requested
    if save_results:
        print(f"\n[SAVE] Saving results to {output_dir}/")
        detailed_file, _ = save_results_to_csv(all_results, dataset_name, target_dim, b_percentage, alpha, k_values, output_dir)
        summary_file, _ = save_summary_report(all_results, dataset_name, target_dim, b_percentage, alpha, k_values, output_dir)
        print(f"[SAVE] Detailed results: {detailed_file}")
        print(f"[SAVE] Summary report: {summary_file}")
        return all_results, detailed_file, summary_file
    else:
        return all_results


if __name__ == "__main__":
    # Example run: Fasttext 1% small case
    dataset_name = "Fasttext"
    train_file = "training_vectors_01pct_Fasttext.npy"
    test_file = "testing_vectors_01pct_Fasttext.npy"
    target_dim = 128
    b_percentage = 1.0
    alpha = 0.1
    k_values = [1, 10, 50]

    main_evaluation_mpad_parallel(dataset_name, train_file, test_file, target_dim, b_percentage, alpha, k_values,
                                  save_results=True, output_dir="Result")


