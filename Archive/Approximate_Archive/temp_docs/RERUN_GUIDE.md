# 重新运行实验指南 (Experiment Re-run Guide)

## 🚨 为什么需要重新运行?

**之前的实现有严重错误**: Ground truth 是在降维后的空间计算的,而不是原始空间。这导致:
- ❌ 只测试了索引方法的精度
- ❌ 没有测试降维方法的质量
- ❌ 所有 Recall@k 值都接近 1.0 (虚高)

**现在已修复**: Ground truth 在原始空间计算,正确评估降维质量。

详见: `GROUND_TRUTH_FIX.md`

---

## 📋 实验清单

### 必须重新运行 ✅

1. **Scalability Test** (Fasttext 1%, 5%, 10%)
   - 预计时间: ~1.5-2 小时
   - 输出: `Result/scalability_fasttext_optimized/`

2. **Large Datasets** (SIFT1M → Fasttext 100% → Deep10M)
   - 预计时间: ~10-15 小时
   - 输出: `Result/large_datasets_*/`

3. **Ablation Studies** (Fasttext, Isolet, PBMC3k, Arcene) - 可选
   - 预计时间: ~2-4 小时/数据集
   - 输出: `Result/ablation_*_optimized/`

---

## 🚀 使用方法

### 方法 1: 运行全部实验 (推荐)

```bash
cd ~/Approximate

# 运行所有实验 (会在每个阶段提示确认)
python rerun_all_experiments.py --all

# 自动运行所有实验 (无提示,适合远程)
nohup python rerun_all_experiments.py --all --no-prompts > rerun.log 2>&1 &
```

### 方法 2: 分步运行

```bash
# 只运行 Scalability test
python rerun_all_experiments.py --scalability

# 只运行大数据集 (SIFT1M → Fasttext 100% → Deep10M)
python rerun_all_experiments.py --large-datasets

# 只运行 Ablation studies
python rerun_all_experiments.py --ablation
```

### 方法 3: 分别运行各个脚本

```bash
# Scalability
python scalability_test_optimized.py

# SIFT1M, Fasttext 100%, Deep10M (按顺序)
python rerun_all_experiments.py --large-datasets

# Ablation (单独数据集)
python ablation_study_optimized.py Fasttext --processes 14
python ablation_study_optimized.py Isolet --processes 14
python ablation_study_optimized.py PBMC3k --processes 14
python ablation_study_optimized.py Arcene --processes 14
```

---

## 🖥️ 远程服务器执行 (推荐)

### 步骤 1: 上传文件和数据

```bash
# 本地执行
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD"

# 上传 Approximate 文件夹
scp -r Approximate/ jiuzhou@er074.utah.cloudlab.us:~/

# 上传数据文件 (如果还没上传)
scp training_vectors_*_Fasttext.npy jiuzhou@er074.utah.cloudlab.us:~/
scp testing_vectors_*_Fasttext.npy jiuzhou@er074.utah.cloudlab.us:~/
scp training_vectors_SIFT1M.npy jiuzhou@er074.utah.cloudlab.us:~/
scp testing_vectors_SIFT1M.npy jiuzhou@er074.utah.cloudlab.us:~/
scp training_vectors_Deep10M.npy jiuzhou@er074.utah.cloudlab.us:~/
scp testing_vectors_Deep10M.npy jiuzhou@er074.utah.cloudlab.us:~/
scp training_vectors_*.npy jiuzhou@er074.utah.cloudlab.us:~/  # 其他数据集
scp testing_vectors_*.npy jiuzhou@er074.utah.cloudlab.us:~/
```

### 步骤 2: 在服务器上运行

```bash
# SSH 到服务器
ssh jiuzhou@er074.utah.cloudlab.us

# 进入 tmux (推荐,避免 SSH 断开)
tmux new -s rerun

# 激活环境
cd ~/Approximate
source ../venv/bin/activate  # 或 conda activate <env>

# 设置环境变量 (优化性能)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_THREADING_LAYER=omp

# 运行全部实验
python rerun_all_experiments.py --all --no-prompts 2>&1 | tee rerun.log

# 分离 tmux: Ctrl+B, 然后按 D
```

### 步骤 3: 监控进度

```bash
# 重新连接到 tmux
tmux attach -t rerun

# 或者查看日志
tail -f ~/Approximate/rerun.log

# 查看 CPU/内存
htop

# 查看 GPU (如果有)
watch -n 1 nvidia-smi
```

### 步骤 4: 下载结果

```bash
# 本地执行
scp -r jiuzhou@er074.utah.cloudlab.us:~/Approximate/Result/ ./Approximate/

# 或者只下载汇总文件
scp jiuzhou@er074.utah.cloudlab.us:~/Approximate/Result/RERUN_SUMMARY.txt ./
scp jiuzhou@er074.utah.cloudlab.us:~/Approximate/Result/*_consolidated/*.csv ./
```

---

## 📊 预期结果

### Recall@k 值的变化

#### 之前 (错误):
```
IndexFlat_kNN_recall@10:  1.00  ← 总是 1.0 (错误!)
HNSWFlat_recall@10:       0.98
IVFPQ_recall@10:          0.92
```

#### 现在 (正确):
```
IndexFlat_kNN_recall@10:  0.85  ← 降低了! (正确,反映降维质量)
HNSWFlat_recall@10:       0.82
IVFPQ_recall@10:          0.75
```

### 预期 Recall 范围 (Fasttext, k=10)

| Method | TD=64 | TD=128 | TD=256 |
|--------|-------|--------|--------|
| MPAD | 0.75-0.90 | 0.85-0.95 | 0.92-0.98 |
| PCA | 0.60-0.80 | 0.75-0.88 | 0.85-0.93 |
| UMAP | 0.65-0.85 | 0.78-0.90 | 0.87-0.94 |
| Random Proj | 0.50-0.70 | 0.65-0.80 | 0.75-0.88 |

---

## 🕐 预计运行时间

### 在 56 核 CPU + GPU 服务器上:

| 实验 | 数据集 | 样本数 | 预计时间 |
|------|--------|--------|----------|
| **Scalability** | Fasttext 1% | 800 | ~15 分钟 |
|  | Fasttext 5% | 4K | ~30 分钟 |
|  | Fasttext 10% | 8K | ~45 分钟 |
| **小计** |  |  | **~1.5 小时** |
| **Large Datasets** | SIFT1M | 1M | ~2 小时 |
|  | Fasttext 100% | 800K | ~4 小时 |
|  | Deep10M | 10M | ~8 小时 |
| **小计** |  |  | **~14 小时** |
| **Ablation** | Fasttext | ~100 实验 | ~2 小时 |
|  | Isolet | ~100 实验 | ~2 小时 |
|  | PBMC3k | ~80 实验 | ~1.5 小时 |
|  | Arcene | ~100 实验 | ~2.5 小时 |
| **小计** |  |  | **~8 小时** |
| **总计** |  |  | **~23-25 小时** |

*如果使用 `--ablation` 跳过 ablation studies: ~15-16 小时*

---

## 📁 输出文件结构

```
Result/
├── scalability_fasttext_optimized/
│   ├── scalability_results_optimized.csv  ← 主要结果
│   └── results_Fasttext_*_TD128_*.csv     ← 详细结果
│
├── large_datasets_SIFT1M/
│   ├── results_SIFT1M_TD64_*.csv
│   └── summary_SIFT1M_TD64_*.txt
│
├── large_datasets_Fasttext_100pct/
│   ├── results_Fasttext_100pct_TD64_*.csv
│   └── summary_Fasttext_100pct_TD64_*.txt
│
├── large_datasets_Deep10M/
│   ├── results_Deep10M_TD64_*.csv
│   └── summary_Deep10M_TD64_*.txt
│
├── large_datasets_consolidated/
│   └── all_large_datasets_results.csv  ← 汇总结果
│
├── ablation_Fasttext_optimized/
│   └── ablation_results_optimized_Fasttext.csv
│
├── ablation_Isolet_optimized/
│   └── ablation_results_optimized_Isolet.csv
│
├── ablation_PBMC3k_optimized/
│   └── ablation_results_optimized_PBMC3k.csv
│
├── ablation_Arcene_optimized/
│   └── ablation_results_optimized_Arcene.csv
│
└── RERUN_SUMMARY.txt  ← 运行总结
```

---

## ✅ 验证清单

重新运行后,检查:

### 1. Recall@k 值是否正确
```bash
# 检查 CSV 文件
head Result/scalability_fasttext_optimized/scalability_results_optimized.csv

# 验证 IndexFlat 的 recall < 1.0
python -c "
import pandas as pd
df = pd.read_csv('Result/scalability_fasttext_optimized/scalability_results_optimized.csv')
print('IndexFlat Recall@10 range:', df['IndexFlat_kNN_recall@10'].min(), '-', df['IndexFlat_kNN_recall@10'].max())
# 应该看到 0.6-0.95 的范围,不是全部 1.0
"
```

### 2. Ground Truth 消息
查看日志,应该看到:
```
================================================================================
COMPUTING GROUND TRUTH IN ORIGINAL SPACE
================================================================================
[INFO] Computing exact kNN on ORIGINAL data...
```

### 3. 文件完整性
```bash
# 检查文件是否生成
ls -lh Result/scalability_fasttext_optimized/
ls -lh Result/large_datasets_*/
ls -lh Result/ablation_*/

# 检查汇总文件
cat Result/RERUN_SUMMARY.txt
```

---

## ⚠️ 故障排除

### 问题 1: OOM (内存不足)
**解决**: 使用更大内存的节点,或者先跳过 Deep10M/Fasttext 100%

### 问题 2: 某个数据集失败
**解决**: 继续运行,脚本会自动跳过失败的数据集

### 问题 3: 进度很慢
**解决**: 检查 CPU 利用率 (`htop`),确保多核在工作

### 问题 4: SSH 断开
**解决**: 使用 `tmux` 或 `nohup`,即使断开也会继续运行

---

## 📞 帮助

- 详细的修复说明: `GROUND_TRUTH_FIX.md`
- 远程执行指南: `REMOTE_EXECUTION_GUIDE.md`
- 优化说明: `MEMORY_AND_PARALLELIZATION_FIX.md`

---

**日期**: 2025-10-26  
**状态**: ✅ 准备就绪  
**重要性**: 🚨 CRITICAL (必须重新运行所有实验)

