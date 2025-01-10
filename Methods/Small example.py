import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

# 1. 数据生成（模拟3维数据）
np.random.seed(42)
X = np.random.multivariate_normal([0, 0, 0], [[10, 3, 1], [3, 8, 2], [1, 2, 6]], 150)

# 2. 数据标准化（中心化）
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# 3. 定义10% DW-PMAD计算函数
def dw_pmad_10(w, X):
    w = w / np.linalg.norm(w)  # 确保方向向量归一化
    projections = X.dot(w)
    n = len(projections)
    abs_diffs = [abs(projections[i] - projections[j]) for i in range(n) for j in range(i + 1, n)]
    abs_diffs.sort()
    top_10_percent = abs_diffs[:int(0.1 * len(abs_diffs))]
    return -np.mean(top_10_percent)  # 取负号用于最小化

# 4. 优化投影方向（最大化10%DW-PMAD）
initial_w1 = np.random.rand(X.shape[1])
initial_w2 = np.random.rand(X.shape[1])
result1 = minimize(dw_pmad_10, initial_w1, args=(X_centered,), method='BFGS')
result2 = minimize(dw_pmad_10, initial_w2, args=(X_centered,), method='BFGS')
optimal_w1 = result1.x / np.linalg.norm(result1.x)
optimal_w2 = result2.x / np.linalg.norm(result2.x)
W_dw_pmad = np.column_stack((optimal_w1, optimal_w2))

# 5. DW-PMAD降维（3D -> 2D）
X_dw_pmad = X_centered.dot(W_dw_pmad)

# 6. PCA降维（3D -> 2D）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_centered)
W_pca = pca.components_.T  # 获取PCA投影方向

# 7. 随机生成10个新数据点
new_points = np.random.multivariate_normal([0, 0, 0], [[10, 3, 1], [3, 8, 2], [1, 2, 6]], 10)
new_points_centered = new_points - X_mean
new_dw_pmad = new_points_centered.dot(W_dw_pmad)
new_pca = pca.transform(new_points_centered)

# 8. 最近邻搜索函数（修复维度问题）
def calculate_accuracy(original_data, reduced_data, new_reduced_data, k=5):
    nbrs_original = NearestNeighbors(n_neighbors=k).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=k).fit(reduced_data)
    total_matches = 0
    for i in range(len(new_reduced_data)):
        original_neighbors = nbrs_original.kneighbors([original_data[i]], return_distance=False)
        reduced_neighbors = nbrs_reduced.kneighbors([new_reduced_data[i]], return_distance=False)
        matches = len(set(original_neighbors[0]) & set(reduced_neighbors[0]))
        total_matches += matches
    accuracy = total_matches / (len(new_reduced_data) * k)
    return accuracy

# 9. 计算准确度
k = 5
accuracy_dw_pmad = calculate_accuracy(X_centered, X_dw_pmad, new_dw_pmad, k)
accuracy_pca = calculate_accuracy(X_centered, X_pca, new_pca, k)

# 10. 可视化
fig = plt.figure(figsize=(18, 6))

# 原始数据（3D）
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.3, label='Original Data')
ax1.quiver(0, 0, 0, *optimal_w1, length=5, color='g', label='DW-PMAD Axis 1')
ax1.quiver(0, 0, 0, *optimal_w2, length=5, color='y', label='DW-PMAD Axis 2')
ax1.quiver(0, 0, 0, *W_pca[:, 0], length=5, color='r', label='PCA Axis 1')
ax1.quiver(0, 0, 0, *W_pca[:, 1], length=5, color='b', label='PCA Axis 2')
ax1.set_title('3D Original Data with Projection Axes')
ax1.legend()

# DW-PMAD降维结果
ax2 = fig.add_subplot(132)
ax2.scatter(X_dw_pmad[:, 0], X_dw_pmad[:, 1], alpha=0.5, color='g', label='DW-PMAD Projected')
ax2.set_title('DW-PMAD 2D Projection')
ax2.legend()

# PCA降维结果
ax3 = fig.add_subplot(133)
ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, color='b', label='PCA Projected')
ax3.set_title('PCA 2D Projection')
ax3.legend()

plt.tight_layout()
plt.show()

# 11. 打印结果
print("DW-PMAD 投影方向:")
print(W_dw_pmad)
print("PCA 投影方向:")
print(W_pca)
print(f"DW-PMAD方法的准确度: {accuracy_dw_pmad * 100:.2f}%")
print(f"PCA方法的准确度: {accuracy_pca * 100:.2f}%")
