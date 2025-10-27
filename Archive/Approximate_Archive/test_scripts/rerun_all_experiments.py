#!/usr/bin/env python3
"""
Complete re-run of all experiments with CORRECT ground truth (from original space).

CRITICAL: All previous results used WRONG ground truth (from reduced space).
This script re-runs everything with the FIXED implementation.

Execution order:
1. Scalability Test: Fasttext 1%, 5%, 10%
2. Large Dataset Evaluations: SIFT1M → Fasttext 100% → Deep10M
3. Ablation Studies: Fasttext, Isolet, PBMC3k, Arcene (optional, can be run separately)

Author: Auto-generated after ground truth fix
Date: 2025-10-26
"""

import os
import sys
import time
import argparse
from datetime import datetime


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*100)
    print(f"{'='*40} {title} {'='*40}")
    print("="*100 + "\n")


def print_step(step_num, total_steps, description):
    """Print a step indicator"""
    print(f"\n{'#'*100}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'#'*100}\n")


def run_scalability_test():
    """Run Fasttext scalability test (1%, 5%, 10%)"""
    print_step(1, 4, "Scalability Test - Fasttext (1%, 5%, 10%)")
    
    print("[INFO] This will test:")
    print("  - Fasttext 1%:  ~800 training samples")
    print("  - Fasttext 5%:  ~4K training samples")
    print("  - Fasttext 10%: ~8K training samples")
    print("  - Methods: 8 (MPAD, PCA, UMAP, RandomProjection, NMF, FeatureAgglomeration, Autoencoder, VAE)")
    print("  - Skipped: Isomap, KernelPCA, LLE (too slow for 5% and 10%)")
    print("  - Expected time: ~1.5-2 hours\n")
    
    input("Press Enter to start scalability test, or Ctrl+C to skip...")
    
    start_time = time.time()
    
    # Import and run
    try:
        from scalability_test_optimized import run_fasttext_scalability_test_optimized
        
        print("\n[RUN] Starting scalability test...")
        results = run_fasttext_scalability_test_optimized()
        
        elapsed = time.time() - start_time
        
        if results is not None:
            print(f"\n[SUCCESS] Scalability test completed in {elapsed/3600:.2f} hours")
            print(f"[OUTPUT] Results saved to: Result/scalability_fasttext_optimized/")
            return True
        else:
            print(f"\n[WARNING] Scalability test completed with warnings")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Scalability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_large_datasets():
    """Run SIFT1M, Fasttext 100%, and Deep10M evaluations"""
    print_step(2, 4, "Large Dataset Evaluations: SIFT1M → Fasttext 100% → Deep10M")
    
    print("[INFO] This will evaluate:")
    print("  1. SIFT1M:      1M samples × 128D → 64D")
    print("  2. Fasttext 100%: 799,910 samples × 300D → 64D")
    print("  3. Deep10M:     10M samples × 96D → 64D")
    print("  - Methods: 8 per dataset (Isomap/KernelPCA/LLE skipped)")
    print("  - Expected time: ~10-15 hours total\n")
    
    input("Press Enter to start large dataset evaluations, or Ctrl+C to skip...")
    
    # We'll create a custom runner that follows the required order
    return run_sift_fasttext_deep_sequence()


def run_sift_fasttext_deep_sequence():
    """Run SIFT1M, then Fasttext 100%, then Deep10M in sequence"""
    import numpy as np
    import pandas as pd
    import gc
    from main_program_optimized import main_evaluation_optimized
    
    all_results = []
    datasets = [
        ("SIFT1M", "training_vectors_SIFT1M.npy", "testing_vectors_SIFT1M.npy"),
        ("Fasttext_100pct", "training_vectors_100pct_Fasttext.npy", "testing_vectors_100pct_Fasttext.npy"),
        ("Deep10M", "training_vectors_Deep10M.npy", "testing_vectors_Deep10M.npy"),
    ]
    
    target_dim = 64
    b_percentage = 1.0
    alpha = 0.1
    k_values = [1, 10, 50]
    skip_methods = ['Isomap', 'KernelPCA', 'LLE']
    
    for dataset_idx, (dataset_tag, train_file, test_file) in enumerate(datasets, 1):
        print_header(f"Dataset {dataset_idx}/3: {dataset_tag}")
        
        # Check files exist
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"[ERROR] Missing files for {dataset_tag}:")
            print(f"  Train: {train_file} (exists: {os.path.exists(train_file)})")
            print(f"  Test:  {test_file} (exists: {os.path.exists(test_file)})")
            print(f"[SKIP] Skipping {dataset_tag}\n")
            continue
        
        output_dir = os.path.join("Result", f"large_datasets_{dataset_tag}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[INFO] Starting evaluation of {dataset_tag}")
        print(f"[INFO] Output directory: {output_dir}")
        
        dataset_start = time.time()
        
        try:
            # Run evaluation
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
            
            # Aggregate results
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
                
                # Extract recall values
                for k in k_values:
                    result_entry[f'IndexFlat_kNN_recall@{k}'] = None
                    for idx_method in ['IndexFlat_kNN', 'HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
                        val = None
                        if idx_method in method_results and isinstance(method_results[idx_method], dict):
                            if k in method_results[idx_method] and isinstance(method_results[idx_method][k], dict):
                                val = method_results[idx_method][k].get('recall', None)
                        result_entry[f'{idx_method}_recall@{k}'] = val
                
                all_results.append(result_entry)
            
            # Clean up memory
            print(f"\n[CLEANUP] Freeing memory after {dataset_tag}...")
            for method_name in list(results.keys()):
                if 'error' not in results[method_name]:
                    results[method_name].pop('X_train_reduced', None)
                    results[method_name].pop('X_test_reduced', None)
                    results[method_name].pop('true_indices', None)
                    for idx_method in ['HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
                        if idx_method in results[method_name]:
                            for k in k_values:
                                if k in results[method_name][idx_method]:
                                    if isinstance(results[method_name][idx_method][k], dict):
                                        results[method_name][idx_method][k].pop('indices', None)
            
            del results
            gc.collect()
            
            dataset_elapsed = time.time() - dataset_start
            print(f"\n[SUCCESS] {dataset_tag} completed in {dataset_elapsed/3600:.2f} hours")
            print(f"[OUTPUT] Results saved to: {output_dir}/\n")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {dataset_tag}: {e}")
            import traceback
            traceback.print_exc()
            print(f"[CONTINUE] Continuing to next dataset...\n")
    
    # Save consolidated results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_dir = "Result/large_datasets_consolidated"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "all_large_datasets_results.csv")
        results_df.to_csv(output_file, index=False)
        
        print_header("Large Dataset Evaluations Complete")
        print(f"[OUTPUT] Consolidated results: {output_file}")
        print(f"[SUMMARY] Processed {len(results_df['dataset'].unique())} datasets, {len(all_results)} total experiments\n")
        
        return True
    else:
        print("\n[WARNING] No results were generated")
        return False


def run_ablation_studies():
    """Run ablation studies for all datasets (optional)"""
    print_step(3, 4, "Ablation Studies (Optional)")
    
    print("[INFO] Ablation studies test different parameter combinations:")
    print("  - Datasets: Fasttext, Isolet, PBMC3k, Arcene")
    print("  - Parameters varied: k, target_dim, b, alpha")
    print("  - Each dataset: ~50-100 experiments")
    print("  - Expected time: ~2-4 hours per dataset")
    print("\n[OPTION] You can run these separately later with:")
    print("  python ablation_study_optimized.py Fasttext --processes 14")
    print("  python ablation_study_optimized.py Isolet --processes 14")
    print("  python ablation_study_optimized.py PBMC3k --processes 14")
    print("  python ablation_study_optimized.py Arcene --processes 14\n")
    
    response = input("Run ablation studies now? (y/N): ").strip().lower()
    
    if response != 'y':
        print("[SKIP] Skipping ablation studies. You can run them later.")
        return True
    
    from ablation_study_optimized import run_ablation_study
    import multiprocessing as mp
    
    datasets = ['Fasttext', 'Isolet', 'PBMC3k', 'Arcene']
    num_processes = max(1, mp.cpu_count() // 2)
    
    print(f"\n[INFO] Running ablation studies with {num_processes} parallel processes\n")
    
    for dataset in datasets:
        print_header(f"Ablation Study: {dataset}")
        
        start_time = time.time()
        
        try:
            results = run_ablation_study(dataset, num_processes=num_processes)
            
            elapsed = time.time() - start_time
            
            if results is not None:
                print(f"\n[SUCCESS] {dataset} ablation completed in {elapsed/3600:.2f} hours")
            else:
                print(f"\n[WARNING] {dataset} ablation completed with warnings")
                
        except Exception as e:
            print(f"\n[ERROR] {dataset} ablation failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"[CONTINUE] Continuing to next dataset...")
    
    return True


def generate_summary_report():
    """Generate a summary report of all completed experiments"""
    print_step(4, 4, "Generating Summary Report")
    
    print("[INFO] Collecting results from:")
    print("  - Result/scalability_fasttext_optimized/")
    print("  - Result/large_datasets_*/")
    print("  - Result/ablation_*/")
    
    summary_lines = []
    summary_lines.append("="*100)
    summary_lines.append("EXPERIMENT RE-RUN SUMMARY")
    summary_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("="*100)
    summary_lines.append("\nIMPORTANT: All experiments use CORRECT ground truth from ORIGINAL space")
    summary_lines.append("Previous results (with wrong ground truth) are INVALID.\n")
    
    # Check what was completed
    completed = []
    
    # Scalability
    scalability_dir = "Result/scalability_fasttext_optimized"
    if os.path.exists(scalability_dir):
        csv_files = [f for f in os.listdir(scalability_dir) if f.endswith('.csv')]
        if csv_files:
            completed.append(f"✓ Scalability Test (Fasttext 1%, 5%, 10%)")
            summary_lines.append(f"\n1. Scalability Test: {scalability_dir}/")
            summary_lines.append(f"   - Files: {', '.join(csv_files)}")
    
    # Large datasets
    for dataset in ['SIFT1M', 'Fasttext_100pct', 'Deep10M']:
        dataset_dir = f"Result/large_datasets_{dataset}"
        if os.path.exists(dataset_dir):
            csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
            if csv_files:
                completed.append(f"✓ {dataset} Evaluation")
                summary_lines.append(f"\n2. {dataset}: {dataset_dir}/")
                summary_lines.append(f"   - Files: {', '.join(csv_files)}")
    
    # Consolidated large datasets
    consol_dir = "Result/large_datasets_consolidated"
    if os.path.exists(consol_dir):
        csv_files = [f for f in os.listdir(consol_dir) if f.endswith('.csv')]
        if csv_files:
            summary_lines.append(f"\n3. Consolidated Large Datasets: {consol_dir}/")
            summary_lines.append(f"   - Files: {', '.join(csv_files)}")
    
    # Ablation studies
    for dataset in ['Fasttext', 'Isolet', 'PBMC3k', 'Arcene']:
        ablation_dir = f"Result/ablation_{dataset}_optimized"
        if os.path.exists(ablation_dir):
            csv_files = [f for f in os.listdir(ablation_dir) if f.endswith('.csv')]
            if csv_files:
                completed.append(f"✓ {dataset} Ablation Study")
                summary_lines.append(f"\n4. {dataset} Ablation: {ablation_dir}/")
                summary_lines.append(f"   - Files: {', '.join(csv_files)}")
    
    summary_lines.append("\n" + "="*100)
    summary_lines.append("COMPLETED EXPERIMENTS:")
    summary_lines.append("="*100)
    for item in completed:
        summary_lines.append(item)
    
    summary_lines.append("\n" + "="*100)
    summary_lines.append("NEXT STEPS:")
    summary_lines.append("="*100)
    summary_lines.append("1. Verify Recall@k values are < 1.0 (typically 0.6-0.95)")
    summary_lines.append("2. Compare different DR methods' recall values")
    summary_lines.append("3. Generate plots with: python plot_scalability_results.py")
    summary_lines.append("4. Analyze consolidated CSV files for paper figures")
    summary_lines.append("="*100)
    
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    
    # Save to file
    summary_file = "Result/RERUN_SUMMARY.txt"
    os.makedirs("Result", exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    print(f"\n[SAVE] Summary saved to: {summary_file}\n")


def main():
    """Main function to orchestrate all experiments"""
    parser = argparse.ArgumentParser(
        description='Re-run all experiments with CORRECT ground truth (from original space)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (will prompt for confirmation at each step)
  python rerun_all_experiments.py --all

  # Run only scalability test
  python rerun_all_experiments.py --scalability

  # Run only large datasets
  python rerun_all_experiments.py --large-datasets

  # Run only ablation studies
  python rerun_all_experiments.py --ablation

  # Skip prompts (run automatically)
  python rerun_all_experiments.py --all --no-prompts
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments (scalability + large datasets + ablation)')
    parser.add_argument('--scalability', action='store_true',
                       help='Run only scalability test (Fasttext 1%%, 5%%, 10%%)')
    parser.add_argument('--large-datasets', action='store_true',
                       help='Run only large datasets (SIFT1M, Fasttext 100%%, Deep10M)')
    parser.add_argument('--ablation', action='store_true',
                       help='Run only ablation studies')
    parser.add_argument('--no-prompts', action='store_true',
                       help='Skip confirmation prompts (for automated runs)')
    
    args = parser.parse_args()
    
    # If no specific option, run all
    if not (args.scalability or args.large_datasets or args.ablation):
        args.all = True
    
    print_header("EXPERIMENT RE-RUN SCRIPT")
    print("[CRITICAL] All previous results used WRONG ground truth (from reduced space)")
    print("[CRITICAL] This script re-runs with CORRECT ground truth (from original space)")
    print("[EXPECT] Recall@k values will be LOWER but CORRECT")
    print("\nSee GROUND_TRUTH_FIX.md for detailed explanation.\n")
    
    if not args.no_prompts:
        response = input("Ready to proceed? (y/N): ").strip().lower()
        if response != 'y':
            print("\n[EXIT] Cancelled by user")
            return
    
    script_start = time.time()
    
    # Run experiments based on arguments
    if args.all or args.scalability:
        success = run_scalability_test()
        if not success and not args.no_prompts:
            response = input("\nScalability test had issues. Continue? (y/N): ").strip().lower()
            if response != 'y':
                print("\n[EXIT] Stopped by user")
                return
    
    if args.all or args.large_datasets:
        success = run_large_datasets()
        if not success and not args.no_prompts:
            response = input("\nLarge dataset evaluations had issues. Continue? (y/N): ").strip().lower()
            if response != 'y':
                print("\n[EXIT] Stopped by user")
                return
    
    if args.all or args.ablation:
        run_ablation_studies()  # Always continues even with issues
    
    # Generate summary
    generate_summary_report()
    
    total_elapsed = time.time() - script_start
    
    print_header("ALL EXPERIMENTS COMPLETE")
    print(f"[TIME] Total elapsed: {total_elapsed/3600:.2f} hours")
    print(f"[OUTPUT] All results saved to Result/ directory")
    print(f"[SUMMARY] See Result/RERUN_SUMMARY.txt for details\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Stopped by user (Ctrl+C)")
        print("[INFO] Partial results may be available in Result/ directory")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

