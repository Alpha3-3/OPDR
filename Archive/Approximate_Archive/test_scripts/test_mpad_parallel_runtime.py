#!/usr/bin/env python3
"""
Test MPAD (parallel variant) runtime on Fasttext at sizes: 1000, 2000, 4000, 8000

- Uses main_program_mpad_parallel.run_mpad_parallel to enable CPU multi-core BLAS inside MPAD
- Target dimension fixed to 150 to match prior scalability tests
"""

import os
import time
import numpy as np

from main_program_mpad_parallel import run_mpad_parallel


def main():
    np.random.seed(1)

    data_file = "training_vectors_01pct_Fasttext.npy"
    if not os.path.exists(data_file):
        print(f"[ERROR] Missing file: {data_file}")
        return

    X_all = np.load(data_file)
    n_total = X_all.shape[0]
    print(f"Loaded Fasttext 1%: {X_all.shape}")

    target_dim = 150
    sizes = [1000, 2000, 4000, 8000]

    results = {}

    for n in sizes:
        if n > n_total:
            print(f"\n[SKIP] n={n}: only {n_total} samples available")
            continue

        idx = np.random.choice(n_total, size=n, replace=False)
        X = X_all[idx]

        print(f"\nTesting MPAD-parallel with n={n}, target_dim={target_dim}")
        t0 = time.time()
        # Use a small test split just to exercise transform; time is dominated by fit
        X_train_red, X_test_red = run_mpad_parallel(X, X[:min(200, n)], target_dim=target_dim,
                                                    b_percentage=1.0, alpha=0.1)
        dt = time.time() - t0
        print(f"[RESULT] n={n}: {dt:.2f}s")
        results[n] = dt

    # Summary
    print("\nSUMMARY (MPAD-parallel)")
    for n in sizes:
        if n in results:
            print(f"n={n}: {results[n]:.2f}s")
        else:
            print(f"n={n}: N/A")


if __name__ == "__main__":
    main()


