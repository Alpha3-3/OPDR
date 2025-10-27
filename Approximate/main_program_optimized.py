#!/usr/bin/env python3
"""
Main evaluation program using optimized MPAD.

This is a modified version of main_program.py that uses the Numba-optimized MPAD
instead of the baseline implementation.

Key differences:
1. Uses MPAD_Optimized from mpad_optimized.py
2. Forces BLAS to single-thread to avoid oversubscription
3. Enables Numba parallel execution

All other methods (baselines and index methods) remain unchanged.
"""

import os
import sys

# Force BLAS to single-thread BEFORE importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Set Numba threading layer
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time

# Import optimized MPAD
from mpad_optimized import MPAD_Optimized

# Import everything else from main_program
from main_program import (
    BaselineMethods,
    IndexMethods,
    get_memory_usage,
    get_peak_memory_usage,
    monitor_memory,
    check_gpu_availability,
    get_gpu_resource,
    save_results_to_csv,
    save_summary_report,
    save_reduced_data_and_results,
)


def evaluate_method_optimized(method_name, method_func, X_train, X_test, target_dim, 
                               index_methods, k_values, true_indices_orig=None):
    """
    Evaluate a single dimensionality reduction method with optimized MPAD.
    
    This is a wrapper around evaluate_method() from main_program.py.
    The only difference is that when method_name is 'MPAD', it uses
    the optimized version which is already defined in run_mpad_optimized().
    
    Args:
        true_indices_orig: Ground truth indices from ORIGINAL space (required)
    """
    from main_program import evaluate_method
    return evaluate_method(method_name, method_func, X_train, X_test, target_dim,
                          index_methods, k_values, true_indices_orig=true_indices_orig)


def main_evaluation_optimized(dataset_name: str,
                              train_file: str,
                              test_file: str,
                              target_dim: int,
                              b_percentage: float,
                              alpha: float,
                              k_values: list,
                              save_results: bool = True,
                              output_dir: str = "Result",
                              skip_methods: list = None):
    """
    Main evaluation function using optimized MPAD.
    
    Args:
        dataset_name: Name of the dataset
        train_file: Path to training data .npy file
        test_file: Path to testing data .npy file
        target_dim: Target dimensionality
        b_percentage: Percentage for top-b pairs in MPAD
        alpha: Orthogonality penalty weight in MPAD
        k_values: List of k values for kNN evaluation
        save_results: Whether to save results to files
        output_dir: Directory to save results
        skip_methods: List of method names to skip (default: None)
    
    Returns:
        all_results: Dictionary of all results
        detailed_file: Path to detailed results CSV (if saved)
        summary_file: Path to summary report (if saved)
    """
    
    if skip_methods is None:
        skip_methods = []
    print(f"\n{'='*80}")
    print(f"MAIN EVALUATION - OPTIMIZED MPAD")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Target Dimension: {target_dim}")
    print(f"MPAD Parameters: b={b_percentage}%, alpha={alpha}")
    print(f"k values: {k_values}")
    if skip_methods:
        print(f"Skipping methods: {', '.join(skip_methods)}")
    print(f"{'='*80}\n")
    
    # Load data
    X_train = np.load(train_file)
    X_test = np.load(test_file)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"\nUsing OPTIMIZED MPAD (Numba parallel)\n")
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Build index methods dict (match main_program.py)
    index_methods = {
        'IndexFlat_kNN': IndexMethods.exact_knn,
        'HNSWFlat': IndexMethods.hnswflat_faiss,
        'IVFPQ': IndexMethods.ivfpq_faiss,
        'IVF_PQR': IndexMethods.ivf_pqr_faiss,
        'IVF_OPQ_PQ': IndexMethods.ivf_opq_pq_faiss,
    }
    
    # ===== CRITICAL: Calculate ground truth in ORIGINAL SPACE (once for all methods) =====
    print(f"\n{'='*80}")
    print("COMPUTING GROUND TRUTH IN ORIGINAL SPACE")
    print(f"{'='*80}")
    print(f"[INFO] Computing exact kNN on ORIGINAL data (train={X_train.shape}, test={X_test.shape})")
    print(f"[INFO] This ground truth will be used to evaluate ALL dimensionality reduction methods")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    true_indices_orig = IndexMethods.exact_knn(X_train, X_test, max(k_values))
    
    gt_time = time.time() - start_time
    end_memory = get_memory_usage()
    gt_memory = end_memory - start_memory
    
    print(f"[OK] Ground truth computed in {gt_time:.4f}s, Memory: {gt_memory:.2f}MB")
    print(f"[INFO] Ground truth shape: {true_indices_orig.shape}")
    print(f"{'='*80}\n")
    
    # Define methods to evaluate
    methods = {
        'MPAD': lambda X_tr, X_te, td: run_mpad_optimized(X_tr, X_te, td, b_percentage, alpha),
        'PCA': lambda X_tr, X_te, td: BaselineMethods.run_pca(X_tr, X_te, td),
        'UMAP': lambda X_tr, X_te, td: BaselineMethods.run_umap(X_tr, X_te, td),
        'Isomap': lambda X_tr, X_te, td: BaselineMethods.run_isomap(X_tr, X_te, td),
        'KernelPCA': lambda X_tr, X_te, td: BaselineMethods.run_kernel_pca(X_tr, X_te, td),
        'RandomProjection': lambda X_tr, X_te, td: BaselineMethods.run_random_projection(X_tr, X_te, td),
        'NMF': lambda X_tr, X_te, td: BaselineMethods.run_nmf(X_tr, X_te, td),
        'LLE': lambda X_tr, X_te, td: BaselineMethods.run_lle(X_tr, X_te, td),
        'FeatureAgglomeration': lambda X_tr, X_te, td: BaselineMethods.run_feature_agglomeration(X_tr, X_te, td),
        'Autoencoder': lambda X_tr, X_te, td: BaselineMethods.run_autoencoder(X_tr, X_te, td),
        'VAE': lambda X_tr, X_te, td: BaselineMethods.run_vae(X_tr, X_te, td),
    }
    
    all_results = {}
    
    # Evaluate each method
    for method_name, method_func in methods.items():
        # Skip methods if requested
        if method_name in skip_methods:
            print(f"\n[SKIP] {method_name} (excluded for this dataset size)")
            continue
            
        try:
            print(f"\n{'='*80}")
            print(f"Evaluating: {method_name}")
            print(f"{'='*80}")
            
            results = evaluate_method_optimized(
                method_name, method_func, X_train, X_test, target_dim,
                index_methods, k_values, true_indices_orig=true_indices_orig
            )
            results['gt_time_orig'] = gt_time  # Store original space GT time
            results['gt_memory_orig'] = gt_memory
            
            all_results[method_name] = results
            
            print(f"\n[{method_name}] Completed successfully")
            print(f"  DR Time: {results['dr_time']:.2f}s")
            print(f"  Memory Used: {results['dr_memory']:.2f} MB")
            
        except Exception as e:
            print(f"\n[{method_name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[method_name] = {'error': str(e)}
    
    # Save results if requested
    detailed_file = None
    summary_file = None
    
    if save_results:
        detailed_file, _ = save_results_to_csv(all_results, dataset_name, target_dim, 
                                               b_percentage, alpha, k_values, output_dir)
        summary_file = save_summary_report(all_results, dataset_name, target_dim,
                                          b_percentage, alpha, output_dir)
        
        print(f"\n{'='*80}")
        print("Results saved:")
        print(f"  Detailed: {detailed_file}")
        print(f"  Summary: {summary_file}")
        print(f"{'='*80}\n")
    
    return all_results, detailed_file, summary_file


def run_mpad_optimized(X_train, X_test, target_dim, b_percentage, alpha):
    """
    Run optimized MPAD with Numba parallel execution.
    
    This function uses MPAD_Optimized which has:
    - Single-threaded BLAS (avoiding oversubscription)
    - Numba parallel binary search for pair counting
    - Parallel equal-layer sampling
    - Parallel gradient coefficient building
    """
    mpad = MPAD_Optimized(b_percentage=b_percentage, alpha=alpha, target_dim=target_dim)
    
    X_train_reduced = mpad.fit_transform(X_train)
    X_test_reduced = mpad.transform(X_test)
    
    return X_train_reduced, X_test_reduced


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("OPTIMIZED MPAD EVALUATION - Example Run")
    print("="*80)
    print("\nConfiguration:")
    print("  - BLAS: Single-threaded")
    print("  - Numba: Parallel enabled")
    print("  - MPAD: Optimized version with parallel binary search")
    print("="*80)
    
    # Test with Fasttext 1%
    results, detail_file, summary_file = main_evaluation_optimized(
        dataset_name="Fasttext_01pct",
        train_file="training_vectors_01pct_Fasttext.npy",
        test_file="testing_vectors_01pct_Fasttext.npy",
        target_dim=128,
        b_percentage=1.0,
        alpha=0.1,
        k_values=[1, 10, 50],
        save_results=True,
        output_dir="Result/optimized_test"
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

