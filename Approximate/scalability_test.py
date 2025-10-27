#!/usr/bin/env python3
"""
Scalability test for Fasttext dataset
Tests 1%, 5%, 10% subsamples with base parameters
"""

import numpy as np
import pandas as pd
from main_program import main_evaluation

def run_fasttext_scalability_test():
    """Run scalability test on Fasttext dataset"""
    
    # Base parameters
    target_dim = 128
    b_percentage = 1.0
    alpha = 0.1
    k_values = [1, 10, 50]
    
    # Subsample ratios
    subsamples = ['01pct', '05pct', '10pct']
    
    results_list = []
    
    print("=" * 60)
    print("Fasttext Scalability Test")
    print("=" * 60)
    
    for subsample in subsamples:
        print(f"\n{'='*60}")
        print(f"Testing {subsample} subsample")
        print(f"{'='*60}\n")
        
        train_file = f"training_vectors_{subsample}_Fasttext.npy"
        test_file = f"testing_vectors_{subsample}_Fasttext.npy"
        
        # Run evaluation
        all_results, detailed_file, summary_file = main_evaluation(
            dataset_name=f"Fasttext_{subsample}",
            train_file=train_file,
            test_file=test_file,
            target_dim=target_dim,
            b_percentage=b_percentage,
            alpha=alpha,
            k_values=k_values,
            save_results=True,
            output_dir="Result/scalability_fasttext"
        )
        
        # Store results
        results_list.append({
            'subsample': subsample,
            'results': all_results,
            'detailed_file': detailed_file,
            'summary_file': summary_file
        })
    
    print(f"\n{'='*60}")
    print("Scalability test completed!")
    print(f"{'='*60}\n")
    
    return results_list

if __name__ == "__main__":
    results = run_fasttext_scalability_test()
    print(f"Total samples tested: {len(results)}")
    print("\nResults summary:")
    for r in results:
        print(f"  {r['subsample']}: {r['detailed_file']}")

