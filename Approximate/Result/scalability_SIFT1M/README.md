# Scalability Analysis - SIFT1M Dataset

## 📁 文件说明

### 数据文件

**注意：这是两个不同参数的实验结果！**

1. **Detailed Results (b=1.0, α=0.1)**
   - `results_SIFT1M_TD64_b1.0_alpha0.1_20251026_224118.csv` - 详细结果数据（每个方法×每个index×每个k的recall）

2. **Consolidated Results (b=0.5, α=0.4)**
   - `consolidated_results.csv` - 合并结果数据（每个方法的runtime和三个k值的recall）

### 脚本文件

1. `plot_detailed_results.py` - 为详细结果生成图表（b=1.0, α=0.1）
2. `plot_consolidated_results.py` - 为合并结果生成图表（b=0.5, α=0.4）

### 生成的图表
每个脚本会生成两张交互式图表（可手动调整保存）

## 🚀 如何运行

### 方法1：Detailed Results (b=1.0, α=0.1)
```bash
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD\Approximate\Result\scalability_SIFT1M"
python plot_detailed_results.py
```

### 方法2：Consolidated Results (b=0.5, α=0.4)
```bash
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD\Approximate\Result\scalability_SIFT1M"
python plot_consolidated_results.py
```

### 输出方式
- 脚本会弹出 **matplotlib 交互式窗口**
- 第一个窗口：Recall@k vs k
- 关闭第一个窗口后显示第二个窗口：Runtime vs Method
- 可以使用工具栏调整和保存图片

## 📊 分析的方法

### Detailed Results (7个方法)
1. **MPAD** - 红色，实心圆点
2. **PCA** - 橙色，空心倒三角
3. **Autoencoder** - 绿色，叉号
4. **RandomProjection** - 灰色，星号
5. **FeatureAgglomeration** - 金色，正方形
6. **UMAP** - 棕色，空心三角
7. **NMF** - 深绿色，六边形

### Consolidated Results (5个方法)
1. **MPAD** - 红色，实心圆点
2. **FeatureAgglomeration** - 金色，正方形
3. **RandomProjection** - 灰色，星号
4. **UMAP** - 棕色，空心三角
5. **NMF** - 深绿色，六边形

### 索引方法（5个）
- kNN (IndexFlat)
- HNSW
- IVFPQ
- IVF-PQR
- IVF-OPQ-PQ

## 📈 图表说明

### 图1: Recall@k vs k
- **横轴**: k值（1, 10, 50）
- **纵轴**: Recall@k
- **子图**: 5个索引方法横向排列
- **用途**: 比较不同降维方法在不同k值下的召回率表现

### 图2: Runtime vs Method
- **横轴**: 降维方法
- **纵轴**: 运行时间（秒）
- **用途**: 比较各降维方法的计算效率

## 🎨 颜色配置

所有样式参考自 `Results/Ablation Study new.py`，使用 `tableau-colorblind10` 配色方案：

| 方法 | 颜色代码 | 颜色名称 |
|------|----------|----------|
| MPAD | `red` | 红色 |
| PCA | `#FF8C00` | 深橙色 |
| Autoencoder | `#32CD32` | 亮绿色 |
| RandomProjection | `#808080` | 灰色 |
| FeatureAgglomeration | `#FFD700` | 金色 |
| UMAP | `#8B4513` | 棕色 |
| NMF | `#006400` | 深绿色 |

## 📝 实验信息

### 实验1: Detailed Results
- **数据集**: SIFT1M
- **向量维度**: 原始128维 → 降至64维
- **参数设置**: **b=1.0, α=0.1**
- **测试日期**: 2025-10-26
- **特点**: 包含每个index方法和每个k值的详细recall数据

### 实验2: Consolidated Results
- **数据集**: SIFT1M
- **向量维度**: 原始128维 → 降至64维
- **参数设置**: **b=0.5, α=0.4**
- **特点**: 只包含k=1, 10, 50三个点的汇总数据

**注意**: 这两个实验使用了不同的参数配置，结果不能直接比较！

