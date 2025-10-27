# Ablation Study 运行时间估算

## 📊 **数据集信息**

基于已有的测试数据，以下是各数据集的规模：

| 数据集 | 训练样本 | 测试样本 | 原始维度 |
|--------|---------|---------|---------|
| Fasttext (1%) | ~8000 | ~2000 | 300 |
| Isolet | ~6237 | ~1560 | 617 |
| PBMC3k | ~2110 | ~528 | 1838 |
| Arcene | ~720 | ~180 | 10000 |

## 🔬 **Ablation Study 配置**

### Fasttext (1%)
- Target Dims: 3 (64, 128, 192)
- b percentages: 5 (0.5, 1.0, 2.0, 4.0, 8.0)
- alphas: 4 (0.05, 0.10, 0.20, 0.40)
- k values: 3 (1, 10, 50)
- **Total experiments**: 3×5×4×3 = **180 experiments**

### Isolet
- Target Dims: 4 (64, 128, 256, 384)
- b percentages: 5
- alphas: 4
- k values: 3
- **Total experiments**: 4×5×4×3 = **240 experiments**

### PBMC3k
- Target Dims: 4 (128, 256, 384, 512)
- b percentages: 5
- alphas: 4
- k values: 3
- **Total experiments**: 4×5×4×3 = **240 experiments**

### Arcene
- Target Dims: 5 (128, 256, 384, 512, 1024)
- b percentages: 5
- alphas: 4
- k values: 3
- **Total experiments**: 5×5×4×3 = **300 experiments**

**总计**: 180 + 240 + 240 + 300 = **960 experiments**

## ⏱️ **单次实验时间估算**

每个实验会运行11种降维方法 + 5种索引方法：
- 11种DR方法: MPAD, PCA, UMAP, Isomap, KernelPCA, RandomProjection, NMF, LLE, FeatureAgglomeration, Autoencoder, VAE
- 5种索引方法: IndexFlat_kNN, HNSWFlat, IVFPQ, IVF_PQR, IVF_OPQ_PQ

### Fasttext (~8000训练样本)

基于测试数据，MPAD是最慢的方法：
- 1000样本: MPAD ~66s
- 2000样本: MPAD ~202s  
- 4000样本: MPAD ~509s
- 8000样本: 预估 ~**900-1100s** (15-18分钟)

**其他10种方法平均耗时**: 约1-50s每个方法

**单次完整实验 (11个方法)**:
- MPAD: ~15分钟
- 其他方法: ~5分钟
- **总计**: ~**20分钟/experiment**

### Isolet (~6237训练样本)
- 估算: ~18分钟/experiment

### PBMC3k (~2110训练样本)
- 估算: ~3分钟/experiment

### Arcene (~720训练样本)
- 估算: ~1分钟/experiment

## 🕐 **总时间估算**

### Fasttext (180 experiments)
- 180 × 20分钟 = **3600分钟** = **60小时**

### Isolet (240 experiments)
- 240 × 18分钟 = **4320分钟** = **72小时**

### PBMC3k (240 experiments)
- 240 × 3分钟 = **720分钟** = **12小时**

### Arcene (300 experiments)
- 300 × 1分钟 = **300分钟** = **5小时**

### **总计**
**60 + 72 + 12 + 5 = 149小时 = 约6.2天**

## 💡 **优化建议**

### 1. 并行化
如果使用8核并行：
- 149小时 ÷ 8 = **约18.6小时 = 不到1天**

### 2. 分段运行
建议按数据集分阶段：
1. Arcene (5小时) - 最快
2. PBMC3k (12小时)
3. Isolet (72小时)
4. Fasttext (60小时) - 最慢

### 3. 在远程机上运行
- 使用tmux保持会话
- 可以设置优先级
- 可以长时间运行

### 4. 采样策略
- 可以先运行部分参数组合测试
- 验证后再运行完整ablation

## 📝 **实际运行建议**

```bash
# 1. 在远程机上创建tmux会话
ssh jiuzhou@amd272.utah.cloudlab.us
tmux new -s ablation

# 2. 激活环境并运行
source venv/bin/activate
cd ~/Approximate
python ablation_study.py

# 3. 分离会话 (Ctrl+B, D)

# 4. 定期检查进度
tmux attach -t ablation
```

## ⚠️ **注意事项**

1. **内存消耗**: 每实验约1-2GB RAM，确保有足够内存
2. **磁盘空间**: 结果文件可能占用数GB
3. **网络稳定性**: 长时间运行需稳定连接
4. **故障恢复**: 建议定期保存中间结果

## 🎯 **结论**

- **单机顺序执行**: 6.2天 (150小时)
- **8核并行**: 约19小时
- **推荐**: 在远程机上并行运行，18-24小时完成
