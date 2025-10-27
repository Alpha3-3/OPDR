#!/usr/bin/env python3
"""Quick single test"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from main_program import MPAD as BaselineMPAD
from mpad_optimized import MPAD_Optimized

# Load data
X_all = np.load("training_vectors_01pct_Fasttext.npy")
np.random.seed(42)
X = X_all[np.random.choice(X_all.shape[0], 500, replace=False)]

print(f"Testing with {X.shape} data, target_dim=10")

# Baseline
print("\n[Baseline MPAD]")
np.random.seed(42)
mpad_b = BaselineMPAD(b_percentage=1.0, alpha=0.1, target_dim=10)
X_red_b = mpad_b.fit_transform(X.copy())
print(f"  Projection axes shape: {mpad_b.projection_axes.shape}")
print(f"  Output shape: {X_red_b.shape}")

# Optimized
print("\n[Optimized MPAD]")
np.random.seed(42)
mpad_o = MPAD_Optimized(b_percentage=1.0, alpha=0.1, target_dim=10)
X_red_o = mpad_o.fit_transform(X.copy())
print(f"  Projection axes shape: {mpad_o.projection_axes.shape}")
print(f"  Output shape: {X_red_o.shape}")

# Compare axes
print("\n[Projection Axes Similarity]")
for i in range(10):
    v_b = mpad_b.projection_axes[:, i]
    v_o = mpad_o.projection_axes[:, i]
    sim = np.abs(np.dot(v_b / np.linalg.norm(v_b), v_o / np.linalg.norm(v_o)))
    status = "[OK]" if sim > 0.95 else "[WARN]" if sim > 0.90 else "[ERROR]"
    print(f"  Axis {i+1}: {sim:.6f} {status}")

# Compare outputs
print("\n[Output Data Comparison]")
for i in range(10):
    col_b = X_red_b[:, i]
    col_o = X_red_o[:, i]
    # Check both positive and negative correlation
    corr = np.corrcoef(col_b, col_o)[0, 1]
    corr_abs = np.abs(corr)
    status = "[OK]" if corr_abs > 0.95 else "[WARN]" if corr_abs > 0.90 else "[ERROR]"
    print(f"  Dim {i+1}: correlation = {corr_abs:.6f} {status}")

print("\n[Statistical Comparison]")
print(f"  Baseline  - mean: {X_red_b.mean():.6f}, std: {X_red_b.std():.6f}")
print(f"  Optimized - mean: {X_red_o.mean():.6f}, std: {X_red_o.std():.6f}")
print(f"  Mean diff: {np.abs(X_red_b.mean() - X_red_o.mean()):.6e}")
print(f"  Std diff: {np.abs(X_red_b.std() - X_red_o.std()):.6e}")

