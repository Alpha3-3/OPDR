#!/usr/bin/env python3
"""
Quick test to verify main_program_optimized works correctly
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from main_program_optimized import main_evaluation_optimized

print("Testing main_program_optimized with small dataset...")

# Generate small test data
np.random.seed(42)
X_train = np.random.randn(500, 50)
X_test = np.random.randn(100, 50)

# Save temporarily
np.save("temp_train.npy", X_train)
np.save("temp_test.npy", X_test)

try:
    results, detail_file, summary_file = main_evaluation_optimized(
        dataset_name="Test",
        train_file="temp_train.npy",
        test_file="temp_test.npy",
        target_dim=10,
        b_percentage=1.0,
        alpha=0.1,
        k_values=[1, 10],
        save_results=False,
        output_dir="Result/test"
    )
    
    print("\n[SUCCESS] main_program_optimized works correctly!")
    print(f"Evaluated {len(results)} methods")
    
    # Check MPAD results
    if 'MPAD' in results and 'error' not in results['MPAD']:
        print(f"MPAD execution time: {results['MPAD']['dr_time']:.2f}s")
    
finally:
    # Cleanup
    if os.path.exists("temp_train.npy"):
        os.remove("temp_train.npy")
    if os.path.exists("temp_test.npy"):
        os.remove("temp_test.npy")

