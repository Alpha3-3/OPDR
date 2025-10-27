#!/usr/bin/env python3
"""
Run full evaluation on SIFT1M, Fasttext 100%, and Deep10M using optimized MPAD.

Settings (dataset-specific):
- SIFT1M: TD=64, b=0.5%, alpha=0.4, k in {1,10,50}
- Fasttext 100%: TD=128, b=4.0%, alpha=0.4, k in {1,10,50}
- Deep10M: TD=64, b=1.0%, alpha=0.1, k in {1,10,50}
- Uses relative paths and saves results under Result/
- Skips slow methods (Isomap, KernelPCA, LLE, tSNE, VAE) for large datasets
- Implements memory cleanup and parallel execution
"""

import os
import numpy as np
import pandas as pd
import gc

from main_program_optimized import main_evaluation_optimized


def ensure_preprocessed():
    """Check presence of preprocessed files; instruct user if missing."""
    needed = [
        ("training_vectors_SIFT1M.npy", "testing_vectors_SIFT1M.npy"),
        ("training_vectors_Deep10M.npy", "testing_vectors_Deep10M.npy"),
    ]
    missing = []
    for tr, te in needed:
        if not os.path.exists(tr) or not os.path.exists(te):
            missing.append((tr, te))
    if missing:
        print("[ERROR] Missing preprocessed files:")
        for tr, te in missing:
            print(f"  - {tr} or {te}")
        print("\nRun preprocessing first:")
        print("  python data_preprocessing.py")
        raise SystemExit(1)


def run_one(dataset_tag, train_file, test_file):
    """
    Run evaluation for one dataset with optimizations:
    - Skip slow methods (Isomap, KernelPCA, LLE, tSNE, VAE)
    - Clean up memory after evaluation
    - Return aggregated results
    """
    # Dataset-specific parameters
    if dataset_tag == "SIFT1M":
        target_dim = 64
        b_percentage = 0.5  # Changed from 1.0% to 0.5%
        alpha = 0.4         # Changed from 0.1 to 0.4
    elif dataset_tag == "Fasttext_100pct":
        target_dim = 128
        b_percentage = 4.0
        alpha = 0.4
    elif dataset_tag == "Deep10M":
        target_dim = 64
        b_percentage = 1.0
        alpha = 0.1
    else:
        # Default parameters
        target_dim = 64
        b_percentage = 1.0
        alpha = 0.1
    
    k_values = [1, 10, 50]
    
    # Skip slow/problematic methods for large datasets
    skip_methods = ['Isomap', 'KernelPCA', 'LLE', 'tSNE', 'VAE']

    output_dir = os.path.join("Result", f"optimized_{dataset_tag}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"RUN {dataset_tag} - OPTIMIZED MPAD")
    print("=" * 80)
    print(f"[INFO] Parameters: TD={target_dim}, b={b_percentage}%, alpha={alpha}")
    print(f"[INFO] k values: {k_values}")
    print(f"[INFO] Skipping slow methods: {', '.join(skip_methods)}")
    print("=" * 80)
    
    try:
        results, detailed, summary = main_evaluation_optimized(
            dataset_name=dataset_tag,
            train_file=train_file,
            test_file=test_file,
            target_dim=target_dim,
            b_percentage=b_percentage,
            alpha=alpha,
            k_values=k_values,
            save_results=True,
            output_dir=output_dir,
            skip_methods=skip_methods
        )
        
        # Aggregate results for CSV export
        results_list = []
        for method_name, method_results in results.items():
            if 'error' in method_results:
                continue
                
            result_entry = {
                'dataset': dataset_tag,
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
        print(f"\n[CLEANUP] Freeing memory for {dataset_tag}...")
        for method_name in list(results.keys()):
            if 'error' not in results[method_name]:
                # Remove large cached arrays
                results[method_name].pop('X_train_reduced', None)
                results[method_name].pop('X_test_reduced', None)
                results[method_name].pop('true_indices', None)
                # Remove indices from each index method
                for idx_method in ['HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
                    if idx_method in results[method_name]:
                        for k in k_values:
                            if k in results[method_name][idx_method]:
                                if isinstance(results[method_name][idx_method][k], dict):
                                    results[method_name][idx_method][k].pop('indices', None)
        
        # Force garbage collection
        gc.collect()
        print(f"[SUCCESS] {dataset_tag} evaluation completed and memory cleaned\n")
        
        return results_list, detailed, summary
        
    except Exception as e:
        print(f"\n[ERROR] Failed to process {dataset_tag}: {e}")
        import traceback
        traceback.print_exc()
        return [], None, None


def main():
    """
    Main function to run evaluations on SIFT1M and Deep10M.
    Aggregates results and saves to consolidated CSV.
    """
    ensure_preprocessed()
    
    all_results = []
    
    # SIFT1M
    print("\n" + "="*80)
    print("PROCESSING SIFT1M")
    print("="*80 + "\n")
    sift_results, sift_detailed, sift_summary = run_one(
        "SIFT1M", 
        "training_vectors_SIFT1M.npy", 
        "testing_vectors_SIFT1M.npy"
    )
    all_results.extend(sift_results)
    
    # Deep10M
    print("\n" + "="*80)
    print("PROCESSING Deep10M")
    print("="*80 + "\n")
    deep_results, deep_detailed, deep_summary = run_one(
        "Deep10M", 
        "training_vectors_Deep10M.npy", 
        "testing_vectors_Deep10M.npy"
    )
    all_results.extend(deep_results)
    
    # Save consolidated results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_dir = "Result/sift_deep_optimized"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "consolidated_results.csv")
        results_df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("ALL EVALUATIONS COMPLETE")
        print("="*80)
        print(f"Consolidated results saved to: {output_file}")
        print(f"Total experiments: {len(all_results)}")
        
        # Print summary statistics
        print("\nDataset Summary:")
        print("-" * 80)
        for dataset in ['SIFT1M', 'Deep10M']:
            dataset_results = results_df[results_df['dataset'] == dataset]
            if not dataset_results.empty:
                n_methods = len(dataset_results)
                avg_dr_time = dataset_results['dr_time'].mean()
                avg_dr_memory = dataset_results['dr_memory'].mean()
                print(f"{dataset:<12} Methods: {n_methods:<2}  "
                      f"Avg DR Time: {avg_dr_time:>8.2f}s  "
                      f"Avg Memory: {avg_dr_memory:>8.2f}MB")
        print("="*80 + "\n")
    else:
        print("\n[WARNING] No results were generated")


if __name__ == "__main__":
    main()


