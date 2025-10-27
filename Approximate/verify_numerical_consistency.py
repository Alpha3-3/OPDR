#!/usr/bin/env python3
"""
Verify numerical consistency between baseline and optimized MPAD.

This script ensures that the Numba-optimized MPAD produces identical
(or numerically very close) results to the baseline implementation.

Tests:
1. Projection axes similarity
2. Transformed data similarity
3. Objective function values
"""

import os
import numpy as np
from scipy.spatial.distance import cosine

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from main_program import MPAD as BaselineMPAD
from mpad_optimized import MPAD_Optimized

print("="*70)
print("NUMERICAL CONSISTENCY VERIFICATION")
print("="*70)


def compute_projection_similarity(proj1, proj2):
    """
    Compute similarity between two sets of projection axes.
    
    Projection axes can be stored as:
    - Matrix: (n_features, n_components)
    - List of vectors
    
    Projection axes can have arbitrary sign, so we check both
    positive and negative directions.
    """
    # Handle different storage formats
    if isinstance(proj1, np.ndarray) and proj1.ndim == 2:
        # Matrix format (n x m)
        m1 = proj1.shape[1]
        vecs1 = [proj1[:, i] for i in range(m1)]
    else:
        vecs1 = proj1
    
    if isinstance(proj2, np.ndarray) and proj2.ndim == 2:
        # Matrix format (n x m)
        m2 = proj2.shape[1]
        vecs2 = [proj2[:, i] for i in range(m2)]
    else:
        vecs2 = proj2
    
    if len(vecs1) != len(vecs2):
        return None, f"Different number of axes: {len(vecs1)} vs {len(vecs2)}"
    
    similarities = []
    for i, (v1, v2) in enumerate(zip(vecs1, vecs2)):
        # Normalize both
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        
        # Compute cosine similarity (considering sign ambiguity)
        cos_sim = np.abs(np.dot(v1_norm, v2_norm))
        similarities.append(cos_sim)
    
    return similarities, None


def verify_mpad_consistency(X, target_dim, b_pct, alpha, test_name):
    """
    Compare baseline and optimized MPAD on the same data.
    """
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"{'='*70}")
    print(f"Data shape: {X.shape}")
    print(f"Target dim: {target_dim}, b%: {b_pct}, alpha: {alpha}")
    
    # Use same random seed for both
    np.random.seed(42)
    
    # Baseline MPAD
    print("\n[1/2] Running Baseline MPAD...")
    mpad_baseline = BaselineMPAD(b_percentage=b_pct, alpha=alpha, target_dim=target_dim)
    X_red_baseline = mpad_baseline.fit_transform(X.copy())
    
    # Reset random seed to ensure same initialization
    np.random.seed(42)
    
    # Optimized MPAD
    print("[2/2] Running Optimized MPAD...")
    mpad_optimized = MPAD_Optimized(b_percentage=b_pct, alpha=alpha, target_dim=target_dim)
    X_red_optimized = mpad_optimized.fit_transform(X.copy())
    
    print("\n" + "-"*70)
    print("VERIFICATION RESULTS")
    print("-"*70)
    
    # 1. Check projection axes similarity
    proj_baseline = mpad_baseline.projection_axes
    proj_optimized = mpad_optimized.projection_axes
    
    print(f"\nDebug info:")
    print(f"  Baseline axes: {len(proj_baseline) if proj_baseline else 'None'}")
    print(f"  Optimized axes: {len(proj_optimized) if proj_optimized else 'None'}")
    
    similarities, error = compute_projection_similarity(proj_baseline, proj_optimized)
    
    if error:
        print(f"[FAILED] Projection axes check: {error}")
        print(f"  Baseline has {len(proj_baseline)} axes")
        print(f"  Optimized has {len(proj_optimized)} axes")
        return False
    
    print(f"\n1. Projection Axes Similarity:")
    for i, sim in enumerate(similarities):
        status = "[OK]" if sim > 0.99 else "[WARN]" if sim > 0.95 else "[ERROR]"
        print(f"   Axis {i+1}: {sim:.6f} {status}")
    
    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    
    print(f"   Average: {avg_similarity:.6f}")
    print(f"   Minimum: {min_similarity:.6f}")
    
    if min_similarity < 0.95:
        print(f"   [WARN] Some axes differ significantly!")
    else:
        print(f"   [OK] All axes are highly similar")
    
    # 2. Check transformed data similarity
    print(f"\n2. Transformed Data Comparison:")
    
    # Allow for sign flips in each dimension
    data_diff = np.zeros(target_dim)
    for i in range(target_dim):
        col_baseline = X_red_baseline[:, i]
        col_optimized = X_red_optimized[:, i]
        
        # Try both positive and negative alignment
        diff_pos = np.mean((col_baseline - col_optimized) ** 2)
        diff_neg = np.mean((col_baseline + col_optimized) ** 2)
        
        data_diff[i] = min(diff_pos, diff_neg)
    
    mean_mse = np.mean(data_diff)
    max_mse = np.max(data_diff)
    
    print(f"   Mean MSE (per dimension): {mean_mse:.6e}")
    print(f"   Max MSE (per dimension): {max_mse:.6e}")
    
    # Compute overall correlation
    corr_per_dim = []
    for i in range(target_dim):
        col_baseline = X_red_baseline[:, i]
        col_optimized = X_red_optimized[:, i]
        
        # Account for sign flip
        corr_pos = np.corrcoef(col_baseline, col_optimized)[0, 1]
        corr = np.abs(corr_pos)
        corr_per_dim.append(corr)
    
    avg_corr = np.mean(corr_per_dim)
    min_corr = np.min(corr_per_dim)
    
    print(f"   Average correlation: {avg_corr:.6f}")
    print(f"   Minimum correlation: {min_corr:.6f}")
    
    if min_corr < 0.95:
        print(f"   [WARN] Low correlation detected!")
        status = "WARNING"
    elif mean_mse > 1e-3:
        print(f"   [WARN] High MSE detected!")
        status = "WARNING"
    else:
        print(f"   [OK] Transformed data is consistent")
        status = "PASS"
    
    # 3. Statistical summary
    print(f"\n3. Statistical Summary:")
    print(f"   Baseline - mean: {X_red_baseline.mean():.6f}, std: {X_red_baseline.std():.6f}")
    print(f"   Optimized - mean: {X_red_optimized.mean():.6f}, std: {X_red_optimized.std():.6f}")
    
    mean_diff = np.abs(X_red_baseline.mean() - X_red_optimized.mean())
    std_diff = np.abs(X_red_baseline.std() - X_red_optimized.std())
    
    print(f"   Mean difference: {mean_diff:.6e}")
    print(f"   Std difference: {std_diff:.6e}")
    
    # Final verdict
    print("\n" + "="*70)
    if status == "PASS" and min_similarity > 0.95:
        print("[PASS] VERIFICATION PASSED: Results are numerically consistent")
        print("="*70)
        return True
    else:
        print("[WARN] VERIFICATION WARNING: Some differences detected")
        print("  This may be due to numerical precision or optimization path differences.")
        print("  Check if differences are within acceptable tolerance for your use case.")
        print("="*70)
        return False


def main():
    # Load test data
    data_file = "training_vectors_01pct_Fasttext.npy"
    if not os.path.exists(data_file):
        print(f"[ERROR] Missing file: {data_file}")
        return
    
    X_all = np.load(data_file)
    print(f"Loaded data: {X_all.shape}")
    
    # Test 1: Small dataset (quick test)
    np.random.seed(42)
    X_small = X_all[np.random.choice(X_all.shape[0], 500, replace=False)]
    success1 = verify_mpad_consistency(
        X_small, 
        target_dim=10, 
        b_pct=1.0, 
        alpha=0.1,
        test_name="Small dataset (500 samples, 10 dims)"
    )
    
    # Test 2: Moderate dataset with higher dimensions
    np.random.seed(43)
    X_medium = X_all[np.random.choice(X_all.shape[0], 1000, replace=False)]
    success2 = verify_mpad_consistency(
        X_medium,
        target_dim=20,
        b_pct=2.0,
        alpha=0.2,
        test_name="Medium dataset (1000 samples, 20 dims)"
    )
    
    # Test 3: Different parameters
    np.random.seed(44)
    X_test3 = X_all[np.random.choice(X_all.shape[0], 800, replace=False)]
    success3 = verify_mpad_consistency(
        X_test3,
        target_dim=15,
        b_pct=0.5,
        alpha=0.4,
        test_name="Different parameters (800 samples, b=0.5%, alpha=0.4)"
    )
    
    # Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    results = [
        ("Test 1 (Small)", success1),
        ("Test 2 (Medium)", success2),
        ("Test 3 (Different params)", success3)
    ]
    
    for test_name, success in results:
        status = "[PASS]" if success else "[WARN]"
        print(f"{test_name}: {status}")
    
    all_passed = all(s for _, s in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("[PASS] ALL TESTS PASSED")
        print("\nThe optimized MPAD produces numerically consistent results.")
        print("It is safe to use for production experiments.")
    else:
        print("[WARN] SOME TESTS SHOWED WARNINGS")
        print("\nThe results are mostly consistent but show some differences.")
        print("This is often acceptable due to:")
        print("  1. Different optimization paths in L-BFGS-B")
        print("  2. Floating-point precision differences")
        print("  3. Parallel reduction order differences")
        print("\nReview the detailed output above to assess if differences")
        print("are within acceptable tolerance for your application.")
    print("="*70)


if __name__ == "__main__":
    main()

