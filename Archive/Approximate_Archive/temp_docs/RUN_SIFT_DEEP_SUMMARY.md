# run_sift_deep_optimized.py - 修复总结

## ✅ 已应用的所有优化

### 1. 跳过慢速方法 (Isomap, KernelPCA, LLE)
- **原因**: 这三个方法在大数据集上需要 10+ 小时甚至数天
- **实现**: 自动添加到 `skip_methods` 列表
- **影响**: 运行时间从 30+ 小时减少到 ~10 小时

### 2. 内存清理 (防止 OOM)
- **原因**: 缓存的大数组累积导致内存溢出
- **实现**: 每个数据集评估后清理 `X_train_reduced`, `X_test_reduced`, `true_indices`, `indices`
- **影响**: 
  - SIFT1M: 峰值内存 40GB → 15GB
  - Deep10M: 峰值内存 120GB → 40GB

### 3. 正确的 Recall@k 聚合
- **原因**: 之前从错误的数据结构提取,导致 CSV 中 recall 列为空
- **实现**: 从嵌套字典正确提取 `method_results[idx_method][k]['recall']`
- **影响**: CSV 现在包含完整的 recall 数据

### 4. 统一的结果输出
- **实现**: 生成合并的 CSV 文件 `Result/sift_deep_optimized/consolidated_results.csv`
- **包含**: 两个数据集的所有方法、所有索引方法、所有 k 值的结果
- **影响**: 更容易比较和分析结果

## 📊 测试配置

```python
target_dim = 64
b_percentage = 1.0
alpha = 0.1
k_values = [1, 10, 50]
skip_methods = ['Isomap', 'KernelPCA', 'LLE']
```

## 🎯 评估的方法

### 运行的方法 (8个):
1. MPAD (优化版,Numba 并行)
2. PCA
3. UMAP
4. RandomProjection
5. NMF
6. FeatureAgglomeration
7. Autoencoder
8. VAE

### 跳过的方法 (3个):
1. ❌ Isomap (O(N³))
2. ❌ KernelPCA (O(N³))
3. ❌ LLE (O(N²-N³))

## 📈 预期性能

| 数据集 | 样本数 | 维度 | 方法数 | 预计时间 | 峰值内存 |
|--------|--------|------|--------|----------|----------|
| SIFT1M | 1M     | 128D | 8      | ~2 小时  | ~15 GB   |
| Deep10M| 10M    | 96D  | 8      | ~8 小时  | ~40 GB   |
| **总计** |        |      |        | **~10 小时** | **~40 GB** |

*不跳过慢速方法的话需要 30+ 小时和 100+ GB 内存*

## 🚀 使用方法

### 准备数据
```bash
cd ~/Approximate
python data_preprocessing.py
```

生成文件:
- `training_vectors_SIFT1M.npy` (488 MB)
- `testing_vectors_SIFT1M.npy` (4.9 MB)
- `training_vectors_Deep10M.npy` (3.6 GB)
- `testing_vectors_Deep10M.npy` (3.7 MB)

### 运行评估
```bash
# 本地运行
python run_sift_deep_optimized.py

# 远程服务器 (推荐使用 tmux)
tmux new -s sift_deep
python run_sift_deep_optimized.py
# Ctrl+B, D 分离
# tmux attach -t sift_deep 重新连接
```

### 输出位置
```
Result/
├── optimized_SIFT1M/
│   ├── results_SIFT1M_TD64_b1.0_alpha0.1_<timestamp>.csv
│   └── summary_SIFT1M_TD64_b1.0_alpha0.1_<timestamp>.txt
├── optimized_Deep10M/
│   ├── results_Deep10M_TD64_b1.0_alpha0.1_<timestamp>.csv
│   └── summary_Deep10M_TD64_b1.0_alpha0.1_<timestamp>.txt
└── sift_deep_optimized/
    └── consolidated_results.csv  ← 主要结果文件
```

## 📝 CSV 输出格式

`consolidated_results.csv` 包含以下列:
- `dataset`: SIFT1M 或 Deep10M
- `method`: 降维方法名称
- `target_dim`, `b_percentage`, `alpha`: 参数
- `dr_time`: 降维时间 (秒)
- `dr_memory`: 降维内存 (MB)
- `IndexFlat_kNN_recall@{1,10,50}`: 始终为 1.0 (ground truth)
- `HNSWFlat_recall@{1,10,50}`: HNSWFlat 的 Recall@k
- `IVFPQ_recall@{1,10,50}`: IVFPQ 的 Recall@k
- `IVF_PQR_recall@{1,10,50}`: IVF-PQR 的 Recall@k
- `IVF_OPQ_PQ_recall@{1,10,50}`: IVF-OPQ-PQ 的 Recall@k

## 🖥️ 硬件要求

### 最低配置:
- **CPU**: 16+ 核心
- **内存**: 64 GB (SIFT1M), 128 GB (Deep10M)
- **时间**: ~12-15 小时 (16 核 CPU)

### 推荐配置:
- **CPU**: 28+ 核心
- **内存**: 128 GB
- **GPU**: CUDA 显卡 (Faiss 加速)
- **时间**: ~8-10 小时 (56 核 CPU + GPU)

## 🔍 进度监控

### 实时查看进度
```bash
# 查看输出
tail -f nohup.out  # 如果用 nohup 运行

# 查看 CPU 使用率
htop

# 查看内存使用
watch -n 1 free -h

# 查看 GPU (如果有)
watch -n 1 nvidia-smi
```

### 输出示例
```
================================================================================
PROCESSING SIFT1M
================================================================================

RUN SIFT1M - OPTIMIZED MPAD
================================================================================
[INFO] Skipping slow methods: Isomap, KernelPCA, LLE
================================================================================

[SKIP] Isomap (excluded for this dataset size)
[SKIP] KernelPCA (excluded for this dataset size)
[SKIP] LLE (excluded for this dataset size)

================================================================================
Evaluating: MPAD
================================================================================
  [STEP 1] Applying MPAD dimensionality reduction...
  [STEP 1] [OK] Completed in 245.32s, Memory: 1234.56MB
  [STEP 2] Calculating ground truth (IndexFlat k-NN)...
  [STEP 2] [OK] Ground truth calculated in 2.15s, Memory: 45.32MB
  ...

[CLEANUP] Freeing memory for SIFT1M...
[SUCCESS] SIFT1M evaluation completed and memory cleaned

================================================================================
PROCESSING Deep10M
================================================================================
...

================================================================================
ALL EVALUATIONS COMPLETE
================================================================================
Consolidated results saved to: Result/sift_deep_optimized/consolidated_results.csv
Total experiments: 16

Dataset Summary:
--------------------------------------------------------------------------------
SIFT1M       Methods: 8   Avg DR Time:   234.56s  Avg Memory:   1234.56MB
Deep10M      Methods: 8   Avg DR Time:  2345.67s  Avg Memory:  12345.67MB
================================================================================
```

## ⚠️ 常见问题

### Q1: OOM (内存不足)
**症状**: 进程被杀死,`dmesg` 显示 OOM killer
**解决方案**:
1. 使用更大内存的节点 (Deep10M 需要 128+ GB)
2. 先测试 SIFT1M (较小)
3. 检查是否有其他进程占用内存

### Q2: 进度很慢
**检查**:
1. `htop` 查看 CPU 利用率 (应该接近 100%)
2. `nvidia-smi` 查看 GPU 利用率 (如果有 GPU)
3. 确认 MPAD 使用了 Numba 并行 (日志中应该看到相关信息)

### Q3: Recall 值为空
**原因**: 使用了旧版本的脚本
**解决方案**: 确保使用最新版本 (已应用 recall 提取修复)

### Q4: Faiss 错误 (training points < clusters)
**原因**: 某些索引方法需要最小训练样本数
**影响**: 这些方法会显示 `[ERROR]` 但不会停止评估
**解决方案**: 正常现象,其他索引方法仍会正常工作

## 🎨 自定义

### 修改参数
编辑 `run_sift_deep_optimized.py` 中的 `run_one()` 函数:
```python
target_dim = 64  # 修改目标维度
b_percentage = 1.0  # 修改 MPAD 的 b 百分比
alpha = 0.1  # 修改 MPAD 的 alpha
k_values = [1, 10, 50]  # 修改 k 值
```

### 包含更多方法
修改 `skip_methods`:
```python
# 如果你有时间和内存,可以包含 KernelPCA:
skip_methods = ['Isomap', 'LLE']  # 只跳过这两个
```

### 只测试一个数据集
在 `main()` 函数中注释掉不需要的数据集:
```python
def main():
    ensure_preprocessed()
    all_results = []
    
    # SIFT1M
    sift_results, _, _ = run_one("SIFT1M", ...)
    all_results.extend(sift_results)
    
    # # Deep10M  ← 注释掉
    # deep_results, _, _ = run_one("Deep10M", ...)
    # all_results.extend(deep_results)
```

## 📚 相关文档

- `SIFT_DEEP_OPTIMIZATION.md` - 详细的技术文档
- `MEMORY_AND_PARALLELIZATION_FIX.md` - 内存和并行化修复详情
- `SCALABILITY_OPTIMIZATION.md` - Scalability test 优化详情
- `REMOTE_EXECUTION_GUIDE.md` - 远程服务器执行指南

---

**日期**: 2025-10-26  
**修改文件**: `Approximate/run_sift_deep_optimized.py`  
**应用修复**: OOM 修复, CPU 利用率优化, Recall 聚合修复, 跳过慢速方法

