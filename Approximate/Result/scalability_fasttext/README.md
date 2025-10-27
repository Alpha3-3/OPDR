# Scalability Analysis - Fasttext Dataset

## 📁 文件说明

### 数据文件
- `results_Fasttext_01pct_TD128_b4.0_alpha0.4_20251026_222242.csv` - 1% 数据集原始结果 (10K vectors)
- `results_Fasttext_05pct_TD128_b4.0_alpha0.4_20251026_224908.csv` - 5% 数据集原始结果 (50K vectors)
- `results_Fasttext_10pct_TD128_b4.0_alpha0.4_20251026_233858.csv` - 10% 数据集原始结果 (100K vectors)
- `scalability_results_optimized.csv` - 合并和优化后的结果数据

### 脚本文件
- `scalability_new_plots.py` - 主脚本：生成可扩展性分析图表

### 生成的图表
- `recall_vs_setsize.png` - Recall@10 vs Set Size（5个索引方法 × 5个降维方法）
- `runtime_ratio_vs_setsize.png` - Runtime Ratio（显示可扩展性）

## 🚀 如何运行

### 方法1：在终端运行
```bash
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD\Approximate\Result\scalability_fasttext"
python scalability_new_plots.py
```

### 方法2：在 VS Code / Cursor 中
- 打开 `scalability_new_plots.py`
- 按 `F5` 或右键选择 "Run Python File in Terminal"

### 输出方式
脚本会弹出 **matplotlib 交互式窗口**，可以：
- ⚙️ 点击工具栏的 **配置子图** 按钮调整边距和间距
- 🔍 使用缩放和平移工具查看细节
- 💾 点击保存按钮导出图片（支持 PNG, PDF, SVG 等格式）
- 关闭第一个窗口后，会自动显示第二个窗口

## 📊 分析的方法

### 降维方法（5个）
1. **MPAD** - 红色，实心圆点
2. **UMAP** - 棕色，空心三角
3. **FeatAgg** (Feature Agglomeration) - 金色，正方形
4. **RandProj** (Random Projection) - 灰色，星号
5. **NMF** - 深绿色，六边形

### 索引方法（5个）
- kNN (IndexFlat)
- HNSW
- IVFPQ
- IVF-PQR
- IVF-OPQ-PQ

## 📈 图表说明

### 图1: Recall@10 vs Set Size
- **横轴**: 数据集大小 (10K, 50K, 100K)
- **纵轴**: Recall@10
- **子图**: 5个索引方法并排显示
- **用途**: 比较不同降维方法在不同数据规模下的召回率

### 图2: Runtime Ratio vs Set Size
- **横轴**: 降维方法
- **纵轴**: 运行时间比率
- **实心柱**: 50K / 10K 的时间比
- **斜线柱**: 100K / 10K 的时间比
- **虚线**: Ratio = 1 参考线
- **用途**: 评估各方法的可扩展性（ratio 越小，可扩展性越好）

## 🎨 颜色配置

所有颜色参考自 `Results/Ablation Study new.py`，使用 `tableau-colorblind10` 配色方案：

| 方法 | 颜色代码 | 颜色名称 |
|------|----------|----------|
| MPAD | `red` | 红色 |
| UMAP | `#8B4513` | 棕色 |
| FeatAgg | `#FFD700` | 金色 |
| RandProj | `#808080` | 灰色 |
| NMF | `#006400` | 深绿色 |

## 📝 关键发现

### 可扩展性排名（从好到差）
1. **UMAP**: 1.56× (50K), 3.55× (100K) - 最佳
2. **MPAD**: 3.13× (50K), 5.28× (100K)
3. **NMF**: 3.83× (50K), 7.91× (100K)
4. **RandProj**: 5.02× (50K), 10.46× (100K)
5. **FeatAgg**: 6.06× (50K), 15.37× (100K) - 最差

### Recall@10 表现
- **MPAD** 在大多数索引方法上保持较高的 Recall
- **NMF** 的 Recall 接近 0，不适用于此任务
- **UMAP** 在可扩展性和 Recall 之间取得良好平衡

