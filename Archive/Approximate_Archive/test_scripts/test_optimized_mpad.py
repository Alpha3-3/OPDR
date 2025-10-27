#!/usr/bin/env python3
"""
Test optimized MPAD (Numba parallel) vs baseline MPAD.

Tests on Fasttext: 1000, 2000, 4000 samples
Target dimension: 150

This test demonstrates the effect of:
1. Forcing BLAS to single-thread (avoiding oversubscription)
2. Using Numba parallel binary search (replacing serial two-pointer)
"""

import os
import time
import gc
import platform
import multiprocessing as mp
import numpy as np

# Display system information
print("="*70)
print("SYSTEM INFORMATION")
print("="*70)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Python: {platform.python_version()}")
print(f"NumPy: {np.__version__}")

# CPU information
cpu_count_logical = mp.cpu_count()
try:
    import psutil
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_freq = psutil.cpu_freq()
    print(f"CPU Logical Cores: {cpu_count_logical}")
    print(f"CPU Physical Cores: {cpu_count_physical}")
    if cpu_freq:
        print(f"CPU Frequency: {cpu_freq.current:.2f} MHz (max: {cpu_freq.max:.2f} MHz)")
except ImportError:
    print(f"CPU Logical Cores: {cpu_count_logical}")
    print(f"CPU Physical Cores: N/A (psutil not available)")

# Environment variables
print("\nENVIRONMENT VARIABLES:")
env_vars = [
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS', 'NUMBA_THREADING_LAYER',
    'OMP_PROC_BIND', 'OMP_PLACES', 'KMP_AFFINITY'
]
for var in env_vars:
    value = os.environ.get(var, 'not set')
    print(f"  {var}: {value}")

print("="*70)

from main_program import MPAD as BaselineMPAD
from mpad_optimized import MPAD_Optimized

# Warmup Numba JIT compilation
print("\n[INFO] Warming up Numba JIT compilation...")
from mpad_optimized import parallel_binary_search_indices
s_warmup = np.sort(np.random.randn(100))
_ = parallel_binary_search_indices(s_warmup, 0.5, 1e-9, 100)
print("[INFO] Numba warmup complete\n")


def test_mpad_variant(MPADClass, X, target_dim, b_pct, alpha, label):
    """Test a specific MPAD implementation."""
    gc.collect()
    
    print(f"\n[{label}]")
    
    # Monitor thread and CPU usage
    try:
        import psutil
        process = psutil.Process()
        
        # Before execution
        threads_before = process.num_threads()
        cpu_affinity = process.cpu_affinity() if hasattr(process, 'cpu_affinity') else None
        
        print(f"  Threads before: {threads_before}")
        if cpu_affinity:
            print(f"  CPU affinity: {len(cpu_affinity)} cores - {sorted(cpu_affinity)[:8]}{'...' if len(cpu_affinity) > 8 else ''}")
        
        # Monitor CPU usage during execution
        cpu_percent_before = process.cpu_percent(interval=0.1)
    except ImportError:
        process = None
        print(f"  (psutil not available for thread monitoring)")
    
    mpad = MPADClass(b_percentage=b_pct, alpha=alpha, target_dim=target_dim)
    
    t0 = time.time()
    X_red = mpad.fit_transform(X)
    dt = time.time() - t0
    
    # After execution
    if process:
        threads_after = process.num_threads()
        cpu_percent_after = process.cpu_percent(interval=0.1)
        
        print(f"  Threads after: {threads_after}")
        print(f"  CPU usage: {cpu_percent_after:.1f}%")
    
    print(f"  Time: {dt:.2f}s")
    print(f"  Output shape: {X_red.shape}")
    print(f"  Throughput: {X.shape[0] / dt:.2f} samples/sec")
    
    return dt, X_red


def main():
    np.random.seed(42)
    
    # Load data
    data_file = "training_vectors_01pct_Fasttext.npy"
    if not os.path.exists(data_file):
        print(f"[ERROR] Missing file: {data_file}")
        return
    
    X_all = np.load(data_file)
    n_total = X_all.shape[0]
    print(f"Loaded Fasttext 1%: {X_all.shape}")
    
    # Test parameters
    target_dim = 150
    b_pct = 1.0
    alpha = 0.1
    sizes = [1000, 2000, 4000]
    
    results = {}
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: Baseline vs Optimized MPAD")
    print("="*70)
    
    for n in sizes:
        if n > n_total:
            print(f"\n[SKIP] n={n}: only {n_total} samples available")
            continue
        
        idx = np.random.choice(n_total, size=n, replace=False)
        X = X_all[idx]
        
        print(f"\n{'='*70}")
        print(f"Testing with n={n}, target_dim={target_dim}")
        print(f"{'='*70}\n")
        
        # Test baseline
        try:
            dt_baseline, _ = test_mpad_variant(
                BaselineMPAD, X, target_dim, b_pct, alpha, "Baseline MPAD"
            )
            results.setdefault(n, {})['baseline'] = dt_baseline
        except Exception as e:
            print(f"[Baseline MPAD] ERROR: {e}")
            results.setdefault(n, {})['baseline'] = None
        
        print()
        
        # Test optimized
        try:
            dt_optimized, _ = test_mpad_variant(
                MPAD_Optimized, X, target_dim, b_pct, alpha, "Optimized MPAD (Numba)"
            )
            results.setdefault(n, {})['optimized'] = dt_optimized
        except Exception as e:
            print(f"[Optimized MPAD] ERROR: {e}")
            results.setdefault(n, {})['optimized'] = None
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Size':<10} {'Baseline (s)':<15} {'Optimized (s)':<15} {'Speedup':<10}")
    print("-"*70)
    
    for n in sizes:
        if n in results:
            baseline = results[n].get('baseline')
            optimized = results[n].get('optimized')
            
            baseline_str = f"{baseline:.2f}" if baseline else "N/A"
            optimized_str = f"{optimized:.2f}" if optimized else "N/A"
            
            if baseline and optimized:
                speedup = baseline / optimized
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            
            print(f"{n:<10} {baseline_str:<15} {optimized_str:<15} {speedup_str:<10}")
    
    print("\n" + "="*70)
    print("NOTES:")
    print("="*70)
    print("1. Optimized MPAD uses Numba parallel binary search")
    print("2. BLAS is forced to single-thread to avoid oversubscription")
    print("3. Best results expected on Linux multi-core servers")
    print("4. Windows may show limited speedup due to thread scheduling")
    print("\nFor best results, run on Linux/WSL2 with:")
    print("  export OMP_PROC_BIND=close")
    print("  export OMP_PLACES=cores")
    print("  export NUMBA_THREADING_LAYER=omp")
    print("="*70)


if __name__ == "__main__":
    main()

