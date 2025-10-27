#!/usr/bin/env python3
"""
Ablation Study (MPAD Parallel Variant)

- Keeps ablation config/flows identical to ablation_study.py
- Uses main_program_mpad_parallel.main_evaluation_mpad_parallel to run evaluations
- 11 DR methods still execute serially per experiment; MPAD runs with CPU multi-core BLAS
"""

import numpy as np
import pandas as pd
import os
import itertools
import multiprocessing as mp
import time

from main_program_mpad_parallel import main_evaluation_mpad_parallel


# Dataset configurations
DATASET_CONFIGS = {
    "Fasttext": {
        "train_files": [
            "training_vectors_01pct_Fasttext.npy"
        ],
        "test_files": [
            "testing_vectors_01pct_Fasttext.npy"
        ],
        "original_dim": 300,
        "target_dims": [64, 128, 192],
        "b_percentages": [0.5, 1.0, 2.0, 4.0, 8.0],
        "alphas": [0.05, 0.10, 0.20, 0.40],
        "k_values": [1, 10, 50]
    },
    "Isolet": {
        "train_files": ["training_vectors_Isolet.npy"],
        "test_files": ["testing_vectors_Isolet.npy"],
        "original_dim": 617,
        "target_dims": [64, 128, 256, 384],
        "b_percentages": [0.5, 1.0, 2.0, 4.0, 8.0],
        "alphas": [0.05, 0.10, 0.20, 0.40],
        "k_values": [1, 10, 50]
    },
    "PBMC3k": {
        "train_files": ["training_vectors_PBMC3k.npy"],
        "test_files": ["testing_vectors_PBMC3k.npy"],
        "original_dim": 1838,
        "target_dims": [128, 256, 384, 512],
        "b_percentages": [0.5, 1.0, 2.0, 4.0, 8.0],
        "alphas": [0.05, 0.10, 0.20, 0.40],
        "k_values": [1, 10, 50]
    },
    "Arcene": {
        "train_files": ["training_vectors_Arcene.npy"],
        "test_files": ["testing_vectors_Arcene.npy"],
        "original_dim": 10000,
        "target_dims": [128, 256, 384, 512, 1024],
        "b_percentages": [0.5, 1.0, 2.0, 4.0, 8.0],
        "alphas": [0.05, 0.10, 0.20, 0.40],
        "k_values": [1, 10, 50]
    }
}


# Baseline parameters for each dataset
BASELINE_PARAMS = {
    "Fasttext": {"k": 10, "target_dim": 128, "b": 1.0, "alpha": 0.1},
    "Isolet": {"k": 10, "target_dim": 256, "b": 1.0, "alpha": 0.1},
    "PBMC3k": {"k": 10, "target_dim": 384, "b": 2.0, "alpha": 0.4},
    "Arcene": {"k": 10, "target_dim": 512, "b": 4.0, "alpha": 0.4}
}


def run_single_experiment(args):
    """Run a single experiment using the MPAD-parallel main evaluation."""
    (dataset_name, train_file, test_file, target_dim, b_percentage, alpha, k_values,
     experiment_type, varied_param, varied_value) = args

    try:
        print(f"Running {experiment_type} for {dataset_name}: {varied_param}={varied_value}")

        # Run evaluation (uses MPAD with CPU multi-core BLAS)
        all_results = main_evaluation_mpad_parallel(
            dataset_name, train_file, test_file, target_dim, b_percentage, alpha, k_values,
            save_results=False
        )
        # main_evaluation_mpad_parallel returns dict if save_results=False
        if isinstance(all_results, tuple):
            all_results = all_results[0]

        # Flatten results for DataFrame
        flattened_results = []

        for method_name, method_results in all_results.items():
            dr_time = method_results.get('dr_time', np.nan)

            for index_name, index_results in method_results.items():
                if index_name in ['dr_time', 'dr_memory', 'gt_time', 'gt_memory']:
                    continue

                for k in k_values:
                    if k in index_results:
                        entry = index_results[k]
                        recall = entry.get('recall', np.nan)
                        search_time = entry.get('time', np.nan)
                        search_memory = entry.get('memory', np.nan)

                        flattened_results.append({
                            'dataset': dataset_name,
                            'train_file': train_file,
                            'test_file': test_file,
                            'experiment_type': experiment_type,
                            'varied_param': varied_param,
                            'varied_value': varied_value,
                            'target_dim': target_dim,
                            'b_percentage': b_percentage,
                            'alpha': alpha,
                            'method': method_name,
                            'index_method': index_name,
                            'k': k,
                            'recall': recall,
                            'dr_time': dr_time,
                            'search_time': search_time,
                            'search_memory': search_memory,
                        })

        return flattened_results

    except Exception as e:
        print(f"Error in experiment {experiment_type} for {dataset_name}: {e}")
        return []


def ablation_study_k(dataset_name, config, baseline_params):
    experiments = []
    for train_file, test_file in zip(config["train_files"], config["test_files"]):
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Missing files: {train_file}, {test_file}")
            continue
        for k in config["k_values"]:
            experiments.append((
                dataset_name, train_file, test_file, baseline_params["target_dim"],
                baseline_params["b"], baseline_params["alpha"], [k],
                "k", "k", k
            ))
    return experiments


def ablation_study_target_dim(dataset_name, config, baseline_params):
    experiments = []
    for train_file, test_file in zip(config["train_files"], config["test_files"]):
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Missing files: {train_file}, {test_file}")
            continue
        for target_dim in config["target_dims"]:
            experiments.append((
                dataset_name, train_file, test_file, target_dim,
                baseline_params["b"], baseline_params["alpha"], config["k_values"],
                "target_dim", "target_dim", target_dim
            ))
    return experiments


def ablation_study_b(dataset_name, config, baseline_params):
    experiments = []
    for train_file, test_file in zip(config["train_files"], config["test_files"]):
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Missing files: {train_file}, {test_file}")
            continue
        for b_percentage in config["b_percentages"]:
            experiments.append((
                dataset_name, train_file, test_file, baseline_params["target_dim"],
                b_percentage, baseline_params["alpha"], config["k_values"],
                "b", "b", b_percentage
            ))
    return experiments


def ablation_study_alpha(dataset_name, config, baseline_params):
    experiments = []
    for train_file, test_file in zip(config["train_files"], config["test_files"]):
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Missing files: {train_file}, {test_file}")
            continue
        for alpha in config["alphas"]:
            experiments.append((
                dataset_name, train_file, test_file, baseline_params["target_dim"],
                baseline_params["b"], alpha, config["k_values"],
                "alpha", "alpha", alpha
            ))
    return experiments


def run_ablation_study(dataset_name: str, study_type: str = "all"):
    config = DATASET_CONFIGS[dataset_name]
    baseline_params = BASELINE_PARAMS[dataset_name]

    print(f"Starting ablation study (MPAD parallel) for {dataset_name}")
    print(f"Baseline parameters: {baseline_params}")

    all_experiments = []

    if study_type in ["all", "k"]:
        all_experiments.extend(ablation_study_k(dataset_name, config, baseline_params))

    if study_type in ["all", "target_dim"]:
        all_experiments.extend(ablation_study_target_dim(dataset_name, config, baseline_params))

    if study_type in ["all", "b"]:
        all_experiments.extend(ablation_study_b(dataset_name, config, baseline_params))

    if study_type in ["all", "alpha"]:
        all_experiments.extend(ablation_study_alpha(dataset_name, config, baseline_params))

    print(f"Total experiments to run: {len(all_experiments)}")

    if len(all_experiments) == 0:
        print("No experiments to run")
        return

    # Run experiments in parallel across CPU cores
    start_time = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_single_experiment, all_experiments)

    # Flatten results
    all_results = []
    for result_list in results:
        all_results.extend(result_list)

    # Convert to DataFrame and save
    df_results = pd.DataFrame(all_results)
    output_file = f"ablation_results_parallel_{dataset_name}.csv"
    df_results.to_csv(output_file, index=False)

    end_time = time.time()
    print(f"\nAblation study (MPAD parallel) completed for {dataset_name}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {output_file}")
    print(f"Total experiments completed: {len(all_results)}")

    return df_results


def run_all_ablation_studies():
    datasets = ["Fasttext", "Isolet", "PBMC3k", "Arcene"]
    for dataset in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"Starting ablation study (MPAD parallel) for {dataset}")
            print(f"{'='*60}")
            results = run_ablation_study(dataset)
            if results is not None and not results.empty:
                print(f"Summary for {dataset}: {len(results)} rows")
        except Exception as e:
            print(f"Error running ablation study for {dataset}: {e}")
            continue


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        study_type = sys.argv[2] if len(sys.argv) > 2 else "all"
        run_ablation_study(dataset_name, study_type)
    else:
        run_all_ablation_studies()


