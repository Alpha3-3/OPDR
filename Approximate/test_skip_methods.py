#!/usr/bin/env python3
"""
Quick test to verify skip_methods functionality.
"""

import numpy as np
from main_program_optimized import main_evaluation_optimized

# Create tiny test data
X_train = np.random.randn(50, 100).astype(np.float32)
X_test = np.random.randn(20, 100).astype(np.float32)

# L2 normalize
X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

np.save("test_train_skip.npy", X_train)
np.save("test_test_skip.npy", X_test)

print("="*80)
print("Testing skip_methods Parameter")
print("="*80)

# Test with skip_methods
results, _, _ = main_evaluation_optimized(
    dataset_name="test_skip",
    train_file="test_train_skip.npy",
    test_file="test_test_skip.npy",
    target_dim=32,
    b_percentage=1.0,
    alpha=0.1,
    k_values=[1, 10],
    save_results=False,
    skip_methods=['Isomap', 'KernelPCA', 'LLE']
)

print("\n" + "="*80)
print("Methods evaluated:")
for method_name in results.keys():
    print(f"  - {method_name}")

skipped = ['Isomap', 'KernelPCA', 'LLE']
for method in skipped:
    if method in results:
        print(f"\n[ERROR] {method} should have been skipped!")
    else:
        print(f"[OK] {method} was correctly skipped")

# Cleanup
import os
os.remove("test_train_skip.npy")
os.remove("test_test_skip.npy")

print("\n" + "="*80)
print("Test complete!")
print("="*80)

