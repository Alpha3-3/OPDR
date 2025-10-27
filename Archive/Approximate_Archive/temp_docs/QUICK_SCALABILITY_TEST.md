# 快速Scalability测试计划

## 测试设置

### 数据
- 来源: Fasttext 1%数据集（约8000样本）
- L2标准化: 已完成
- 样本大小: 1000, 2000, 4000, 8000

### 方法
11种降维方法：MPAD, PCA, UMAP, Isomap, KernelPCA, RandomProjection, NMF, LLE, FeatureAgglomeration, Autoencoder, VAE

### 参数
- 目标维度: 150
- MPAD参数: b=1.0%, alpha=0.1

## 执行命令

```bash
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD\Approximate"
python quick_scalability_test_simple.py
```

## 预期结果格式

```
SUMMARY: Runtime (seconds)
================================================================================
Method                           1000      2000      4000      8000
--------------------------------------------------------------------------------
MPAD                             XXX.XX    XXX.XX    XXX.XX    XXX.XX
PCA                               XX.XX     XX.XX     XX.XX     XX.XX
UMAP                              XX.XX     XX.XX     XX.XX     XX.XX
Isomap                            XX.XX     XX.XX     XX.XX     XX.XX
KernelPCA                         XX.XX     XX.XX     XX.XX     XX.XX
RandomProjection                    X.XX      X.XX      X.XX      X.XX
NMF                                XX.XX     XX.XX     XX.XX     XX.XX
LLE                                XX.XX     XX.XX     XX.XX     XX.XX
FeatureAgglomeration               XX.XX     XX.XX     XX.XX     XX.XX
Autoencoder                        XX.XX     XX.XX     XX.XX     XX.XX
VAE                                XX.XX     XX.XX     XX.XX     XX.XX
```

## 预估运行时间

基于之前MPAD测试（8000样本约5分钟），预估总运行时间：
- 1000样本: ~45分钟（11个方法 × ~4分钟）
- 2000样本: ~60分钟
- 4000样本: ~75分钟  
- 8000样本: ~90分钟
- 总计: 约4-5小时

## 建议

1. **在远程机上运行**：使用tmux会话，避免网络中断
2. **简化版本**：可以先测试几个方法（如MPAD, PCA, UMAP）
3. **分段测试**：一次测试一个样本大小
