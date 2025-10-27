#!/usr/bin/env python3
"""Quick verification test for debugging"""

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
X = X_all[np.random.choice(X_all.shape[0], 200, replace=False)]

print(f"Testing with {X.shape} data")

# Baseline
print("\nBaseline MPAD:")
np.random.seed(42)
mpad_b = BaselineMPAD(b_percentage=1.0, alpha=0.1, target_dim=5)
X_red_b = mpad_b.fit_transform(X.copy())
print(f"  Projection axes: {len(mpad_b.projection_axes)}")
print(f"  Output shape: {X_red_b.shape}")

# Optimized
print("\nOptimized MPAD:")
np.random.seed(42)
mpad_o = MPAD_Optimized(b_percentage=1.0, alpha=0.1, target_dim=5)
X_red_o = mpad_o.fit_transform(X.copy())
print(f"  Projection axes: {len(mpad_o.projection_axes)}")
print(f"  Output shape: {X_red_o.shape}")

# Compare
if len(mpad_b.projection_axes) == len(mpad_o.projection_axes):
    print("\n[OK] Both have same number of axes")
    for i in range(len(mpad_b.projection_axes)):
        v_b = mpad_b.projection_axes[i]
        v_o = mpad_o.projection_axes[i]
        sim = np.abs(np.dot(v_b / np.linalg.norm(v_b), v_o / np.linalg.norm(v_o)))
        print(f"  Axis {i+1} similarity: {sim:.6f}")
else:
    print(f"\n[ERROR] Different number of axes!")
    print(f"  Baseline: {len(mpad_b.projection_axes)}")
    print(f"  Optimized: {len(mpad_o.projection_axes)}")

