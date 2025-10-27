#!/usr/bin/env python3
"""
Scalability test using OPTIMIZED MPAD (Numba parallel)

Tests Fasttext dataset with 1%, 5%, 10% subsamples using base parameters.
Uses the optimized MPAD implementation for faster execution.
"""

import os

# Force BLAS single-thread BEFORE any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import numpy as np
import pandas as pd
from main_program_optimized import main_evaluation_optimized

def run_fasttext_scalability_test_optimized():
    """Run scalability test on Fasttext dataset with optimized MPAD"""
    
    # Base parameters (updated for better results)
    target_dim = 128
    b_percentage = 4.0  # Changed from 1.0% to 4.0%
    alpha = 0.4         # Changed from 0.1 to 0.4
    k_values = [1, 10, 50]
    
    # Subsample ratios
    subsamples = ['01pct', '05pct', '10pct']
    
    results_list = []
    
    print("=" * 80)
    print("Fasttext Scalability Test - OPTIMIZED MPAD")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Target Dimension: {target_dim}")
    print(f"  b percentage: {b_percentage}%")
    print(f"  alpha: {alpha}")
    print(f"  k values: {k_values}")
    print(f"  Subsamples: {subsamples}")
    print("\nOptimization:")
    print("  - BLAS: Single-threaded (avoiding oversubscription)")
    print("  - MPAD: Numba parallel binary search")
    print("  - Expected: 5-15x speedup on multi-core systems")
    print("=" * 80)
    
    for subsample in subsamples:
        print(f"\n{'='*80}")
        print(f"Testing {subsample} subsample")
        print(f"{'='*80}\n")
        
        train_file = f"training_vectors_{subsample}_Fasttext.npy"
        test_file = f"testing_vectors_{subsample}_Fasttext.npy"
        
        # Check if files exist
        if not os.path.exists(train_file):
            print(f"[WARNING] File not found: {train_file}")
            print(f"  Skipping {subsample} subsample")
            continue
        
        if not os.path.exists(test_file):
            print(f"[WARNING] File not found: {test_file}")
            print(f"  Skipping {subsample} subsample")
            continue
        
        # Determine which methods to skip based on dataset size
        skip_methods = []
        if subsample in ['05pct', '10pct']:
            skip_methods = ['Isomap', 'KernelPCA', 'LLE']
            print(f"\n[INFO] Large dataset ({subsample}): skipping slow methods: {', '.join(skip_methods)}\n")
        
        # Run evaluation with optimized MPAD
        try:
            all_results, detailed_file, summary_file = main_evaluation_optimized(
                dataset_name=f"Fasttext_{subsample}",
                train_file=train_file,
                test_file=test_file,
                target_dim=target_dim,
                b_percentage=b_percentage,
                alpha=alpha,
                k_values=k_values,
                save_results=True,
                output_dir="Result/scalability_fasttext_optimized",
                skip_methods=skip_methods
            )
            
            # Store results BEFORE clearing memory
            for method_name, method_results in all_results.items():
                if 'error' in method_results:
                    continue
                result_entry = {
                    'subsample': subsample,
                    'method': method_name,
                    'target_dim': target_dim,
                    'b_percentage': b_percentage,
                    'alpha': alpha,
                    'dr_time': method_results.get('dr_time', None),
                    'dr_memory': method_results.get('dr_memory', None),
                }
                # Extract recall values from nested structure
                # IMPORTANT: Now using ground truth from ORIGINAL space, so IndexFlat recall < 1.0
                for k in k_values:
                    for idx_method in ['IndexFlat_kNN', 'HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
                        val = None
                        if idx_method in method_results and isinstance(method_results[idx_method], dict):
                            if k in method_results[idx_method] and isinstance(method_results[idx_method][k], dict):
                                val = method_results[idx_method][k].get('recall', None)
                        result_entry[f'{idx_method}_recall@{k}'] = val
                results_list.append(result_entry)
            
            # Clean up memory-heavy cached data
            print(f"\n[CLEANUP] Freeing memory for {subsample}...")
            for method_name in list(all_results.keys()):
                if 'error' not in all_results[method_name]:
                    # Remove large cached arrays
                    all_results[method_name].pop('X_train_reduced', None)
                    all_results[method_name].pop('X_test_reduced', None)
                    all_results[method_name].pop('true_indices', None)
                    # Remove indices from each index method
                    for idx_method in ['HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
                        if idx_method in all_results[method_name]:
                            for k in k_values:
                                if k in all_results[method_name][idx_method]:
                                    all_results[method_name][idx_method][k].pop('indices', None)
            
            # Force garbage collection
            import gc
            del all_results
            gc.collect()
            
            print(f"[SUCCESS] Completed {subsample} subsample")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {subsample} subsample: {e}")
            import traceback
            traceback.print_exc()
    
    # Save consolidated results
    if results_list:
        results_df = pd.DataFrame(results_list)
        output_dir = "Result/scalability_fasttext_optimized"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "scalability_results_optimized.csv")
        results_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print("SCALABILITY TEST COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"Total experiments: {len(results_list)}")
        
        # Print summary statistics for MPAD
        mpad_results = results_df[results_df['method'] == 'MPAD']
        if not mpad_results.empty:
            print(f"\nMPAD Performance Summary:")
            print(f"{'Subsample':<12} {'Train Size':<12} {'DR Time (s)':<15} {'Memory (MB)':<15}")
            print("-" * 80)
            for subsample in subsamples:
                sub_data = mpad_results[mpad_results['subsample'] == subsample]
                if not sub_data.empty:
                    # Get data size
                    train_file = f"training_vectors_{subsample}_Fasttext.npy"
                    if os.path.exists(train_file):
                        train_size = np.load(train_file).shape[0]
                    else:
                        train_size = "N/A"
                    
                    dr_time = sub_data['dr_time'].values[0]
                    dr_memory = sub_data['dr_memory'].values[0]
                    print(f"{subsample:<12} {train_size:<12} {dr_time:<15.2f} {dr_memory:<15.2f}")
        
        print(f"{'='*80}\n")
        
        return results_df
    else:
        print("\n[WARNING] No results were generated")
        return None


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("SCALABILITY TEST - OPTIMIZED MPAD")
    print("="*80)
    print("\nThis script tests the optimized MPAD across different dataset sizes.")
    print("Expected speedup: 5-15x compared to baseline MPAD on multi-core systems.")
    print("="*80 + "\n")
    
    results = run_fasttext_scalability_test_optimized()
    
    if results is not None:
        print("\n[SUCCESS] Scalability test completed successfully")
        sys.exit(0)
    else:
        print("\n[ERROR] Scalability test failed")
        sys.exit(1)

