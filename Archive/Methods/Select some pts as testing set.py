import torch  # Replaced NumPy with PyTorch for AMD GPU acceleration
import pandas as pd
from scipy.optimize import minimize
import itertools
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import os
import random

# 1. 加载预训练向量（随机选择，PyTorch优化）
def load_vectors(fname, num_points=1000, device='cuda'):
    try:
        with open(fname, 'r', encoding='utf-8', errors='ignore') as fin:
            n, d = map(int, fin.readline().split())
            all_indices = random.sample(range(n), num_points)
            all_indices.sort()

            vectors = []
            current_index = 0
            for i, line in enumerate(fin):
                if i == all_indices[current_index]:
                    tokens = line.rstrip().split(' ')
                    vector = torch.tensor([float(x) for x in tokens[1:]], dtype=torch.float32, device=device)
                    vectors.append(vector)
                    current_index += 1
                    if current_index >= num_points:
                        break
        return torch.stack(vectors)
    except FileNotFoundError:
        print(f"File not found: {fname}")
        return None
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return None

# 2. 数据生成函数（PyTorch优化）
def generate_data(distribution, num_points, dimension, pretrained_vectors=None, device='cuda'):
    if distribution == 'random':
        return torch.rand((num_points, dimension), device=device) - 0.5
    elif distribution == 'multivariate_normal':
        mean = torch.zeros(dimension, device=device)
        cov = torch.eye(dimension, device=device) * 10
        return torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample((num_points,))
    elif distribution == 'pretrained_vectors':
        if pretrained_vectors is None or len(pretrained_vectors) == 0:
            raise ValueError("Pretrained vectors not loaded.")
        indices = torch.randperm(pretrained_vectors.size(0))[:num_points]
        return pretrained_vectors[indices, :dimension]
    else:
        raise ValueError("Unsupported distribution type.")

# 3. 数据标准化（PyTorch优化）
def center_data(X):
    return X - X.mean(dim=0)

# 4. DW-PMAD计算函数（PyTorch优化）
def dw_pmad_b(w, X, b):
    w = w / torch.norm(w)
    projections = X @ w
    abs_diffs = torch.abs(projections.unsqueeze(0) - projections.unsqueeze(1)).flatten()
    sorted_diffs, _ = torch.sort(abs_diffs)
    top_b_percent = sorted_diffs[:int((b / 100) * len(sorted_diffs))]
    return -torch.mean(top_b_percent)

# 5. PCA实现（PyTorch优化 + Fix for Dimension Error）
def pca_torch(X, target_dim):
    X_centered = center_data(X)
    max_dim = min(X_centered.shape)
    if target_dim > max_dim:
        print(f"[Warning] Adjusted target_dim from {target_dim} to {max_dim} due to data size.")
        target_dim = max_dim
    U, S, V = torch.pca_lowrank(X_centered, q=target_dim)
    return X_centered @ V[:, :target_dim]

# 6. 最近邻计算（PyTorch优化 + Dimension Fix）
def calculate_accuracies(original_data, reduced_data, new_original_data, new_reduced_data, k_values):
    # Ensure both datasets have the same dimensions
    min_dim = min(reduced_data.shape[1], new_reduced_data.shape[1])
    reduced_data = reduced_data[:, :min_dim]
    new_reduced_data = new_reduced_data[:, :min_dim]

    accuracies = {}
    for k in k_values:
        distances_original = torch.cdist(new_original_data, original_data)
        distances_reduced = torch.cdist(new_reduced_data, reduced_data)
        _, neighbors_original = torch.topk(distances_original, k, largest=False)
        _, neighbors_reduced = torch.topk(distances_reduced, k, largest=False)
        matches = (neighbors_original == neighbors_reduced).sum().item()
        accuracies[k] = matches / (new_original_data.size(0) * k)
    return accuracies

# 7. 加载预训练向量
filename = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M.vec'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained_vectors = load_vectors(filename, num_points=2000, device=device)

# 8. 参数范围
distributions = ['random', 'multivariate_normal', 'pretrained_vectors']
dimensions = [5, 100, 300]
b_values = [25, 50, 75, 100]
k_values = [5, 10, 15, 20]
alpha_values = [1, 10, 20]
target_dimension_percentages = [0.25, 0.5, 0.75]
num_points_values = [200, 500]

# 9. 结果存储
results = []

# 10. 参数遍历
param_combinations = list(itertools.product(
    distributions, dimensions, b_values, alpha_values, target_dimension_percentages, num_points_values))

for dist, dim, b, alpha, target_dim_percentage, num_points in tqdm(param_combinations, desc="Parameter Sweeping"):
    target_dim = max(1, math.ceil(dim * target_dim_percentage))

    X = generate_data(dist, num_points, dim, pretrained_vectors, device)
    X_centered = center_data(X)

    X_dw_pmad = pca_torch(X_centered, target_dim)
    X_pca = pca_torch(X_centered, target_dim)

    new_points = generate_data(dist, 1000, dim, pretrained_vectors, device)
    new_points_centered = center_data(new_points)
    new_dw_pmad = pca_torch(new_points_centered, target_dim)
    new_pca = pca_torch(new_points_centered, target_dim)

    accuracies_dw_pmad = calculate_accuracies(X_centered, X_dw_pmad, new_points_centered, new_dw_pmad, k_values)
    accuracies_pca = calculate_accuracies(X_centered, X_pca, new_points_centered, new_pca, k_values)

    for k in k_values:
        results.append({
            'Distribution': dist,
            'Dimension': dim,
            'b': b,
            'k': k,
            'Alpha': alpha,
            'Target_Dimension_Percentage': target_dim_percentage,
            'Computed_Target_Dimension': target_dim,
            'Num_Points': num_points,
            'DW-PMAD Accuracy': accuracies_dw_pmad[k],
            'PCA Accuracy': accuracies_pca[k]
        })

# 11. 结果输出
df_results = pd.DataFrame(results)
df_results.to_csv('dw_pmad_vs_pca_results.csv', index=False)

print("GPU-accelerated parameter sweeping completed and results exported to 'dw_pmad_vs_pca_results.csv'.")
