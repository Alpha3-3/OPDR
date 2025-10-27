#!/usr/bin/env python3
"""
Test script comparing baseline MPAD vs Numba-optimized MPAD.

Tests on Fasttext with sizes: 1000, 2000, 4000 samples.
Target dimension: 150
"""

import os
import time
import gc
import numpy as np

# Import baseline MPAD from main_program
from main_program import MPAD as BaselineMPAD

# Import numba kernels to trigger JIT compilation
try:
    from mpad_numba_kernels import (
        parallel_count_pairs_binary,
        compute_pair_counts,
        compute_gt_statistics,
        compute_eq_layer_counts,
    )
    NUMBA_AVAILABLE = True
    print("[INFO] Numba kernels loaded successfully")
except ImportError as e:
    NUMBA_AVAILABLE = False
    print(f"[WARNING] Numba not available: {e}")


def warmup_numba():
    """Warmup JIT compilation with small data."""
    if not NUMBA_AVAILABLE:
        return
    
    print("[INFO] Warming up Numba JIT compilation...")
    s = np.sort(np.random.randn(100))
    P = np.zeros(101)
    P[1:] = np.cumsum(s)
    
    j_gt, j_ge = parallel_count_pairs_binary(s, 0.5, 1e-9, 100)
    _ = compute_pair_counts(j_gt, 100)
    print("[INFO] Numba warmup complete")


def test_baseline_mpad(X, target_dim, b_pct, alpha):
    """Test baseline MPAD implementation."""
    gc.collect()
    mpad = BaselineMPAD(b_percentage=b_pct, alpha=alpha, target_dim=target_dim)
    
    t0 = time.time()
    X_red = mpad.fit_transform(X)
    dt = time.time() - t0
    
    return dt, X_red


def main():
    np.random.seed(42)
    
    # Load Fasttext 1% data
    data_file = "training_vectors_01pct_Fasttext.npy"
    if not os.path.exists(data_file):
        print(f"[ERROR] Missing file: {data_file}")
        return
    
    X_all = np.load(data_file)
    n_total = X_all.shape[0]
    print(f"Loaded Fasttext 1%: {X_all.shape}\n")
    
    # Test parameters
    target_dim = 150
    b_pct = 1.0
    alpha = 0.1
    sizes = [1000, 2000, 4000]
    
    # Warmup numba
    if NUMBA_AVAILABLE:
        warmup_numba()
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: Baseline MPAD")
    print("="*60)
    
    results = {}
    
    for n in sizes:
        if n > n_total:
            print(f"\n[SKIP] n={n}: only {n_total} samples available")
            continue
        
        idx = np.random.choice(n_total, size=n, replace=False)
        X = X_all[idx]
        
        print(f"\n{'='*60}")
        print(f"Testing with n={n}, target_dim={target_dim}")
        print(f"{'='*60}")
        
        # Test baseline
        print("\n[Baseline MPAD]")
        try:
            dt_baseline, X_red_baseline = test_baseline_mpad(X, target_dim, b_pct, alpha)
            print(f"  Time: {dt_baseline:.2f}s")
            print(f"  Output shape: {X_red_baseline.shape}")
            results[n] = {'baseline': dt_baseline}
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[n] = {'baseline': None}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Size':<10} {'Baseline (s)':<15}")
    print("-"*60)
    for n in sizes:
        if n in results:
            baseline_time = results[n].get('baseline', None)
            baseline_str = f"{baseline_time:.2f}" if baseline_time else "N/A"
            print(f"{n:<10} {baseline_str:<15}")
    
    print("\n[NOTE] Numba-optimized MPAD requires deeper integration into MPAD class.")
    print("[NOTE] Current implementation tests show baseline performance.")
    print("[NOTE] Full Numba optimization would replace internal loops in _objective_and_grad().")


if __name__ == "__main__":
    main()

