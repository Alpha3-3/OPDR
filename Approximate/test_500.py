#!/usr/bin/env python3
"""测试500样本"""

import os
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_method(method_name, X, target_dim, b_percentage=1.0, alpha=0.1):
    """测试单个方法的耗时"""
    from main_program import MPAD
    from sklearn.decomposition import PCA as skPCA
    from sklearn.random_projection import GaussianRandomProjection
    
    try:
        start_time = time.time()
        
        if method_name == 'MPAD':
            mpad = MPAD(b_percentage=b_percentage, alpha=alpha, target_dim=target_dim)
            X_reduced = mpad.fit_transform(X)
        elif method_name == 'PCA':
            pca = skPCA(n_components=target_dim, random_state=1)
            X_reduced = pca.fit_transform(X)
        elif method_name == 'RandomProjection':
            rp = GaussianRandomProjection(n_components=target_dim, random_state=1)
            X_reduced = rp.fit_transform(X)
        else:
            return None
        
        elapsed_time = time.time() - start_time
        return {'time': elapsed_time, 'shape': X_reduced.shape}
    except Exception as e:
        print(f"  [ERROR]: {e}")
        return None

# 运行测试
np.random.seed(1)
print("Loading data...")
train_data = np.load("training_vectors_01pct_Fasttext.npy")
print(f"Loaded {len(train_data)} samples\n")

n = 500
target_dim = 150
methods = ['PCA', 'RandomProjection', 'MPAD']

print("=" * 60)
print(f"Testing with {n} samples, target_dim={target_dim}")
print("=" * 60)

indices = np.random.choice(len(train_data), size=n, replace=False)
X = train_data[indices]
print(f"Sample shape: {X.shape}\n")

results = {}

for method_name in methods:
    print(f"Testing {method_name}... ", end='', flush=True)
    result = test_method(method_name, X, target_dim)
    if result:
        results[method_name] = result
        print(f"{result['time']:.2f}s")
    else:
        print("FAILED")

print("\n" + "=" * 60)
print("RESULTS: Runtime (seconds)")
print("=" * 60)
print(f"{'Method':<20} {'Time (s)':>10}")
print("-" * 60)

for method in methods:
    if method in results:
        print(f"{method:<20} {results[method]['time']:>10.2f}")
    else:
        print(f"{method:<20} {'N/A':>10}")

print("\nTest completed!")
