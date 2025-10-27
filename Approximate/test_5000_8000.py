#!/usr/bin/env python3
"""测试5000,6000,7000,8000样本（带内存清理）"""

import os
import numpy as np
import time
import warnings
import gc

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_method(method_name, X, target_dim, b_percentage=1.0, alpha=0.1):
    """测试单个方法的耗时"""
    from main_program import MPAD
    from sklearn.decomposition import PCA as skPCA
    from sklearn.random_projection import GaussianRandomProjection
    import gc
    
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
        
        # 清理内存
        del X_reduced
        gc.collect()
        
        return {'time': elapsed_time, 'shape': (X.shape[0], target_dim)}
    except Exception as e:
        print(f"  [ERROR]: {e}")
        return None

# 运行测试
np.random.seed(1)
print("Loading data...")
train_data = np.load("training_vectors_01pct_Fasttext.npy")
print(f"Loaded {len(train_data)} samples\n")

# 释放内存
train_data_copy = train_data.copy()
del train_data
gc.collect()

sample_sizes = [5000, 6000, 7000, 8000]
target_dim = 150
methods = ['PCA', 'RandomProjection', 'MPAD']

print("=" * 60)
print(f"Testing with target_dim={target_dim}")
print("=" * 60)

results = {}

for n in sample_sizes:
    print(f"\n{'Testing with ' + str(n) + ' samples':60}")
    print("-" * 60)
    
    if n > len(train_data_copy):
        print(f"Not enough data (have {len(train_data_copy)}, need {n})")
        continue
        
    indices = np.random.choice(len(train_data_copy), size=n, replace=False)
    X = train_data_copy[indices].copy()
    
    results[n] = {}
    
    for method_name in methods:
        print(f"{method_name:<20}... ", end='', flush=True)
        
        # 清理内存
        gc.collect()
        
        try:
            result = test_method(method_name, X, target_dim)
            if result:
                results[n][method_name] = result
                print(f"{result['time']:.2f}s")
            else:
                print("FAILED")
        except Exception as e:
            print(f"ERROR: {e}")
        
        # 清理内存
        gc.collect()

# 打印结果
print("\n" + "=" * 60)
print("RESULTS: Runtime (seconds)")
print("=" * 60)
print(f"{'Method':<20} {'5000':>10} {'6000':>10} {'7000':>10} {'8000':>10}")
print("-" * 60)

for method in methods:
    row = f"{method:<20}"
    for n in sample_sizes:
        if n in results and method in results[n] and results[n][method]['time']:
            row += f"{results[n][method]['time']:>10.2f}"
        else:
            row += f"{'N/A':>10}"
    print(row)
    
print("\nTest completed!")
