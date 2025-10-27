#!/usr/bin/env python3
"""
快速Scalability测试：测试各baseline方法的耗时
只测试降维步骤，不测试Recall等
"""

import os
import sys
import numpy as np
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_fasttext_original():
    """从原始Fasttext文件加载数据"""
    print("Loading Fasttext original data...")
    
    vec_file_path = '../Dataset processing and testing/Fasttext/data/wiki-news-300d-1M.vec'
    
    if not os.path.exists(vec_file_path):
        print(f"File not found: {vec_file_path}")
        return None
    
    vectors = []
    with open(vec_file_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 301:
                continue
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:301]])
            vectors.append(vector)
    
    vectors = np.array(vectors)
    print(f"Loaded {len(vectors)} vectors from original file")
    return vectors

def random_sample(data, n):
    """随机选择n个点"""
    if n >= len(data):
        return data
    indices = np.random.choice(len(data), size=n, replace=False)
    return data[indices]

def l2_normalize(X):
    """L2标准化"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # 避免除以0
    return X / norms

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

def run_scalability_test():
    """运行Scalability测试"""
    print("=" * 80)
    print("Quick Scalability Test: Testing Runtime for Different Sample Sizes")
    print("=" * 80)
    
    # 设置随机种子
    np.random.seed(1)
    
    # 加载原始数据
    original_data = load_fasttext_original()
    if original_data is None:
        print("[ERROR] Failed to load Fasttext data")
        return
    
    print(f"\nOriginal data shape: {original_data.shape}")
    
    # L2标准化
    print("\nL2 normalizing data...")
    normalized_data = l2_normalize(original_data)
    print(f"Normalized data shape: {normalized_data.shape}")
    
    # 测试不同样本大小
    sample_sizes = [1000, 2000, 4000, 8000]
    target_dim = 150
    
    print(f"\nTarget dimension: {target_dim}")
    print(f"Sample sizes to test: {sample_sizes}")
    
    # 定义所有方法
    methods = [
        'MPAD', 'PCA', 'UMAP', 'Isomap', 'KernelPCA', 'RandomProjection',
        'NMF', 'LLE', 'FeatureAgglomeration', 'Autoencoder', 'VAE'
    ]
    
    results = {}
    
    for n in sample_sizes:
        print(f"\n{'='*80}")
        print(f"Testing with {n} samples")
        print(f"{'='*80}")
        
        # 随机采样
        X = random_sample(normalized_data, n)
        print(f"Sampled data shape: {X.shape}")
        
        results[n] = {}
        
        for method_name in methods:
            print(f"\nTesting {method_name}...")
            
            try:
                result = test_method(method_name, X, target_dim)
                if result:
                    results[n][method_name] = result
                    print(f"  [OK] Time: {result['time']:.2f}s, Shape: {result['shape']}")
                else:
                    results[n][method_name] = {'time': None, 'shape': None}
            except Exception as e:
                print(f"  [ERROR] {method_name}: {e}")
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
            if method in results[n] and results[n][method]['time'] is not None:
                row += f"{results[n][method]['time']:>10.2f}"
            else:
                row += f"{'N/A':>10}"
        print(row)
    
    print("\n" + "=" * 80)
    print("Test completed!")
    return results

if __name__ == "__main__":
    results = run_scalability_test()
    
    # 计算加速比
    print("\n" + "=" * 80)
    print("Speedup Analysis (relative to MPAD)")
    print("=" * 80)
    print(f"{'Method':<25} {'1000→2000':>12} {'2000→4000':>12} {'4000→8000':>12}")
    print("-" * 80)
    
    for method in ['PCA', 'UMAP', 'Isomap', 'KernelPCA', 'RandomProjection', 'NMF', 'LLE', 'FeatureAgglomeration', 'Autoencoder', 'VAE']:
        if method not in ['MPAD']:
            row = f"{method:<25}"
            for n in [1000, 2000, 4000]:
                try:
                    if method in results[n] and results[n][method]['time'] and \
                       'MPAD' in results[n] and results[n]['MPAD']['time']:
                        speedup = results[n]['MPAD']['time'] / results[n][method]['time']
                        row += f"{speedup:>12.2f}x"
                    else:
                        row += f"{'N/A':>12}"
                except:
                    row += f"{'N/A':>12}"
            print(row)
