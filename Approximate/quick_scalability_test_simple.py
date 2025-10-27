#!/usr/bin/env python3
"""
快速Scalability测试：使用已经预处理好的数据
测试1000, 2000, 4000, 8000个点的耗时
"""

import os
import sys
import numpy as np
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_method(method_name, X, target_dim, b_percentage=1.0, alpha=0.1):
    """测试单个方法的耗时"""
    from main_program import (
        MPAD, PCA, UMAP, Isomap, KernelPCA, RandomProjection,
        NMF, LLE, FeatureAgglomeration, Autoencoder, VAE
    )
    
    # 导入方法
    methods = {
        'MPAD': lambda: MPAD(b_percentage=b_percentage, alpha=alpha, target_dim=target_dim),
        'PCA': lambda: PCA(target_dim=target_dim),
        'UMAP': lambda: UMAP(target_dim=target_dim),
        'Isomap': lambda: Isomap(target_dim=target_dim),
        'KernelPCA': lambda: KernelPCA(target_dim=target_dim),
        'RandomProjection': lambda: RandomProjection(target_dim=target_dim),
        'NMF': lambda: NMF(target_dim=target_dim),
        'LLE': lambda: LLE(target_dim=target_dim),
        'FeatureAgglomeration': lambda: FeatureAgglomeration(target_dim=target_dim),
        'Autoencoder': lambda: Autoencoder(target_dim=target_dim),
        'VAE': lambda: VAE(target_dim=target_dim),
    }
    
    if method_name not in methods:
        return None
    
    try:
        start_time = time.time()
        model = methods[method_name]()
        X_reduced = model.fit_transform(X)
        elapsed_time = time.time() - start_time
        
        return {
            'time': elapsed_time,
            'shape': X_reduced.shape
        }
    except Exception as e:
        print(f"  [ERROR] {method_name}: {e}")
        return None

def run_test():
    """运行测试"""
    print("=" * 80)
    print("Quick Scalability Test")
    print("=" * 80)
    
    # 设置随机种子
    np.random.seed(1)
    
    # 加载数据（Fasttext 1%，已经L2标准化过）
    print("\nLoading preprocessed data...")
    train_data = np.load("training_vectors_01pct_Fasttext.npy")
    print(f"Loaded {len(train_data)} samples")
    
    # 测试不同样本大小
    sample_sizes = [1000, 2000, 4000, 8000]
    target_dim = 150
    
    print(f"\nTarget dimension: {target_dim}")
    
    methods = ['MPAD', 'PCA', 'UMAP', 'Isomap', 'KernelPCA', 'RandomProjection',
               'NMF', 'LLE', 'FeatureAgglomeration', 'Autoencoder', 'VAE']
    
    results = {}
    
    for n in sample_sizes:
        print(f"\n{'='*80}")
        print(f"Testing with {n} samples")
        print(f"{'='*80}")
        
        if n > len(train_data):
            print(f"Not enough data (have {len(train_data)}, need {n})")
            continue
            
        # 随机采样
        indices = np.random.choice(len(train_data), size=n, replace=False)
        X = train_data[indices]
        print(f"Sampled data shape: {X.shape}")
        
        results[n] = {}
        
        for method_name in methods:
            print(f"Testing {method_name}...", end=' ')
            
            try:
                result = test_method(method_name, X, target_dim)
                if result:
                    results[n][method_name] = result
                    print(f"[OK] {result['time']:.2f}s")
                else:
                    results[n][method_name] = {'time': None, 'shape': None}
                    print(f"[FAILED]")
            except Exception as e:
                print(f"[ERROR]: {e}")
                results[n][method_name] = {'time': None, 'shape': None}
    
    # 打印汇总结果
    print("\n" + "=" * 80)
    print("SUMMARY: Runtime (seconds)")
    print("=" * 80)
    print(f"{'Method':<25} {'1000':>10} {'2000':>10} {'4000':>10} {'8000':>10}")
    print("-" * 80)
    
    for method in methods:
        row = f"{method:<25}"
        for n in sample_sizes:
            if n in results and method in results[n] and results[n][method]['time'] is not None:
                row += f"{results[n][method]['time']:>10.2f}"
            else:
                row += f"{'N/A':>10}"
        print(row)
    
    return results

if __name__ == "__main__":
    results = run_test()
    print("\nTest completed!")
