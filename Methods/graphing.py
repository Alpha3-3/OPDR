import matplotlib.pyplot as plt
import numpy as np

# 数据准备
k_values = [3, 5, 10, 20, 50, 100]
metrics = ["PCA", "RandomProjection", "TopSTD", "TopAbsDiff", "GroupByAbsDiffMean"]
data = [
    [47.67, 34.33, 39.67, 38.67, 41.67],
    [62, 41.2, 51, 52.6, 50],
    [69.5, 45.3, 60.3, 60.3, 59.9],
    [74.6, 51.2, 65.4, 65, 64.8],
    [77.82, 54.34, 66.36, 66.6, 67.32],
    [79.73, 55.6, 68.23, 68.24, 68.79]
]

# 转置数据，使指标为x轴，k的数值为y轴
data_transposed = np.array(data).T

# 画图
plt.figure(figsize=(10, 6))
for i, k in enumerate(k_values):
    plt.plot(metrics, data_transposed[:, i], marker='o', label=f'k={k}')

# 图表设置
plt.title("Performance Comparison Across Metrics (n=150)")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# 显示图表
plt.show()
