#!/usr/bin/env python3
"""
Ablation study using OPTIMIZED MPAD (Numba parallel)

Tests different parameter combinations for MPAD and baseline methods.
Uses the optimized MPAD implementation for significantly faster execution.
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
import itertools
import multiprocessing as mp
from main_program_optimized import main_evaluation_optimized
import time


# Dataset configurations
DATASET_CONFIGS = {
    "Fasttext": {
        "train_files": ["training_vectors_01pct_Fasttext.npy"],
        "test_files": ["testing_vectors_01pct_Fasttext.npy"],
        "original_dim": 300,
        "base_params": {
            "target_dim": 128,
            "b_percentage": 1.0,
            "alpha": 0.1
        },
        "target_dims": [64, 128, 192],
        "b_percentages": [0.5, 1.0, 2.0, 4.0, 8.0],
        "alphas": [0.05, 0.10, 0.20, 0.40],
        "k_values": [1, 10, 50]
    },
    "Isolet": {
        "train_files": ["training_vectors_Isolet.npy"],
        "test_files": ["testing_vectors_Isolet.npy"],
        "original_dim": 617,
        "base_params": {
            "target_dim": 256,
            "b_percentage": 1.0,
            "alpha": 0.1
        },
        "target_dims": [64, 128, 256, 384],
        "b_percentages": [0.5, 1.0, 2.0, 4.0, 8.0],
        "alphas": [0.05, 0.10, 0.20, 0.40],
        "k_values": [1, 10, 50]
    },
    "PBMC3k": {
        "train_files": ["training_vectors_PBMC3k.npy"],
        "test_files": ["testing_vectors_PBMC3k.npy"],
        "original_dim": 1838,
        "base_params": {
            "target_dim": 384,
            "b_percentage": 2.0,
            "alpha": 0.4
        },
        "target_dims": [128, 256, 384, 512],
        "b_percentages": [0.5, 1.0, 2.0, 4.0, 8.0],
        "alphas": [0.05, 0.10, 0.20, 0.40],
        "k_values": [1, 10, 50]
    },
    "Arcene": {
        "train_files": ["training_vectors_Arcene.npy"],
        "test_files": ["testing_vectors_Arcene.npy"],
        "original_dim": 10000,
        "base_params": {
            "target_dim": 512,
            "b_percentage": 4.0,
            "alpha": 0.4
        },
        "target_dims": [128, 256, 384, 512, 1024],
        "b_percentages": [0.5, 1.0, 2.0, 4.0, 8.0],
        "alphas": [0.05, 0.10, 0.20, 0.40],
        "k_values": [1, 10, 50]
    }
}


def run_single_experiment(args):
    """Run a single ablation experiment"""
    (dataset_name, train_file, test_file, target_dim, 
     b_percentage, alpha, k_values, exp_id, total_exps) = args
    
    print(f"\n{'='*80}")
    print(f"Experiment {exp_id}/{total_exps}: {dataset_name}")
    print(f"  TD={target_dim}, b={b_percentage}%, alpha={alpha}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        
        all_results, detailed_file, summary_file = main_evaluation_optimized(
            dataset_name=f"{dataset_name}_ablation",
            train_file=train_file,
            test_file=test_file,
            target_dim=target_dim,
            b_percentage=b_percentage,
            alpha=alpha,
            k_values=k_values,
            save_results=False,  # We'll consolidate results ourselves
            output_dir=f"Result/ablation_{dataset_name}_optimized",
            skip_methods=['Isomap', 'KernelPCA', 'LLE', 'tSNE', 'VAE']  # Skip slow/problematic methods
        )
        
        elapsed = time.time() - start_time
        
        # Clean up memory-heavy cached data BEFORE returning
        import gc
        for method_name in list(all_results.keys()):
            if 'error' not in all_results[method_name]:
                all_results[method_name].pop('X_train_reduced', None)
                all_results[method_name].pop('X_test_reduced', None)
                all_results[method_name].pop('true_indices', None)
                # Remove indices from each index method
                for idx_method in ['HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
                    if idx_method in all_results[method_name]:
                        for k in k_values:
                            if k in all_results[method_name][idx_method]:
                                if isinstance(all_results[method_name][idx_method][k], dict):
                                    all_results[method_name][idx_method][k].pop('indices', None)
        gc.collect()
        
        print(f"\n[SUCCESS] Experiment {exp_id}/{total_exps} completed in {elapsed:.1f}s (memory cleaned)")
        
        return {
            'dataset': dataset_name,
            'target_dim': target_dim,
            'b_percentage': b_percentage,
            'alpha': alpha,
            'results': all_results,
            'success': True
        }
        
    except Exception as e:
        print(f"\n[ERROR] Experiment {exp_id}/{total_exps} failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'dataset': dataset_name,
            'target_dim': target_dim,
            'b_percentage': b_percentage,
            'alpha': alpha,
            'error': str(e),
            'success': False
        }


def run_ablation_study(dataset_name, num_processes=None):
    """
    Run ablation study for a specific dataset using optimized MPAD.
    
    Args:
        dataset_name: Name of dataset ('Fasttext', 'Isolet', 'PBMC3k', 'Arcene')
        num_processes: Number of parallel processes (default=None for auto: cpu_count//2)
    """
    
    # Auto-detect optimal process count if not specified
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() // 2)
        print(f"[INFO] Auto-detected {mp.cpu_count()} CPUs, using {num_processes} processes")
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name]
    base_params = config['base_params']
    
    print("="*80)
    print(f"ABLATION STUDY - {dataset_name} - OPTIMIZED MPAD")
    print("="*80)
    print(f"\nBase parameters:")
    print(f"  Target Dim: {base_params['target_dim']}")
    print(f"  b percentage: {base_params['b_percentage']}%")
    print(f"  alpha: {base_params['alpha']}")
    print(f"\nOptimization:")
    print(f"  - BLAS: Single-threaded")
    print(f"  - MPAD: Numba parallel (5-15x faster)")
    print(f"  - Processes: {num_processes}")
    print("="*80)
    
    # Generate all parameter combinations
    experiments = []
    
    # Vary target_dim (fix b, alpha)
    for td in config['target_dims']:
        experiments.append((
            dataset_name, config['train_files'][0], config['test_files'][0],
            td, base_params['b_percentage'], base_params['alpha'],
            config['k_values']
        ))
    
    # Vary b_percentage (fix td, alpha)
    for b in config['b_percentages']:
        if b != base_params['b_percentage']:  # Avoid duplicate
            experiments.append((
                dataset_name, config['train_files'][0], config['test_files'][0],
                base_params['target_dim'], b, base_params['alpha'],
                config['k_values']
            ))
    
    # Vary alpha (fix td, b)
    for a in config['alphas']:
        if a != base_params['alpha']:  # Avoid duplicate
            experiments.append((
                dataset_name, config['train_files'][0], config['test_files'][0],
                base_params['target_dim'], base_params['b_percentage'], a,
                config['k_values']
            ))
    
    # Add experiment IDs
    total_exps = len(experiments)
    experiments_with_ids = [
        exp + (i+1, total_exps) for i, exp in enumerate(experiments)
    ]
    
    print(f"\nTotal experiments: {total_exps}")
    print(f"Processing mode: {'Parallel' if num_processes > 1 else 'Sequential'}")
    
    # Adjust Numba threads for multi-process scenario
    if num_processes > 1:
        # Reduce per-process thread count to avoid oversubscription
        total_cores = mp.cpu_count()
        threads_per_process = max(1, total_cores // num_processes)
        os.environ["NUMBA_NUM_THREADS"] = str(threads_per_process)
        print(f"[INFO] Multi-process mode: {num_processes} processes × {threads_per_process} Numba threads = {num_processes * threads_per_process} total threads")
    else:
        # Single process: use all cores for Numba
        os.environ["NUMBA_NUM_THREADS"] = str(mp.cpu_count())
        print(f"[INFO] Single-process mode: using {mp.cpu_count()} Numba threads")
    
    # Run experiments
    all_experiment_results = []
    
    if num_processes > 1:
        print(f"\nRunning experiments in parallel with {num_processes} processes...")
        with mp.Pool(processes=num_processes) as pool:
            all_experiment_results = pool.map(run_single_experiment, experiments_with_ids)
    else:
        print(f"\nRunning experiments sequentially...")
        for exp_args in experiments_with_ids:
            result = run_single_experiment(exp_args)
            all_experiment_results.append(result)
    
    # Consolidate results
    results_list = []
    
    for exp_result in all_experiment_results:
        if exp_result['success']:
            for method_name, method_results in exp_result['results'].items():
                if 'error' not in method_results:
                    result_entry = {
                        'dataset': exp_result['dataset'],
                        'target_dim': exp_result['target_dim'],
                        'b_percentage': exp_result['b_percentage'],
                        'alpha': exp_result['alpha'],
                        'method': method_name,
                        'dr_time': method_results.get('dr_time', None),
                        'dr_memory': method_results.get('dr_memory', None),
                    }
                    
                    # Add recall@k results from nested structure
                    # IMPORTANT: Now using ground truth from ORIGINAL space, so IndexFlat recall < 1.0
                    for k in config['k_values']:
                        for idx_method in ['IndexFlat_kNN', 'HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
                            val = None
                            if idx_method in method_results and isinstance(method_results[idx_method], dict):
                                if k in method_results[idx_method] and isinstance(method_results[idx_method][k], dict):
                                    val = method_results[idx_method][k].get('recall', None)
                            result_entry[f'{idx_method}_recall@{k}'] = val
                    
                    results_list.append(result_entry)
    
    # Save results
    if results_list:
        results_df = pd.DataFrame(results_list)
        output_dir = f"Result/ablation_{dataset_name}_optimized"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"ablation_results_optimized_{dataset_name}.csv")
        results_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print("ABLATION STUDY COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"Total successful experiments: {len(results_list)}")
        print(f"{'='*80}\n")
        
        return results_df
    else:
        print("\n[WARNING] No results were generated")
        return None


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study with optimized MPAD')
    parser.add_argument('dataset', choices=['Fasttext', 'Isolet', 'PBMC3k', 'Arcene'],
                       help='Dataset to run ablation study on')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: auto-detect, use cpu_count//2)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"ABLATION STUDY - {args.dataset} - OPTIMIZED MPAD")
    print("="*80)
    print("\nUsing optimized MPAD with Numba parallel execution.")
    print("Expected speedup: 5-15x compared to baseline MPAD.")
    print("="*80 + "\n")
    
    results = run_ablation_study(args.dataset, num_processes=args.processes)
    
    if results is not None:
        print(f"\n[SUCCESS] Ablation study for {args.dataset} completed successfully")
        sys.exit(0)
    else:
        print(f"\n[ERROR] Ablation study for {args.dataset} failed")
        sys.exit(1)

