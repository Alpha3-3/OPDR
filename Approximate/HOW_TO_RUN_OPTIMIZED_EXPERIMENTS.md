# 如何使用优化版MPAD运行实验

## 📋 概述

我已经创建了使用优化版MPAD的新脚本，它们与原始脚本并行存在，不会覆盖原有的实验设置。

## 🆕 新建的文件

### 1. 核心文件

- **`main_program_optimized.py`** - 使用优化MPAD的主评估程序
- **`mpad_optimized.py`** - 优化的MPAD实现（已有）

### 2. 实验脚本

- **`scalability_test_optimized.py`** - Scalability测试（优化版）
- **`ablation_study_optimized.py`** - Ablation study（优化版）

### 3. 原始文件（保持不变）

- `main_program.py` - 原始主程序
- `scalability_test.py` - 原始scalability测试
- `ablation_study.py` - 原始ablation study

## ⚙️ 环境设置

### 在本地Windows

优化版已经内置了环境变量设置，无需手动设置。

### 在远程Linux服务器

```bash
# 设置环境变量（推荐在~/.bashrc中添加）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_THREADING_LAYER=omp
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

## 🚀 运行方法

### 1. Scalability Test（可扩展性测试）

测试Fasttext在1%, 5%, 10%数据量下的性能。

#### 本地运行

```bash
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD\Approximate"
python scalability_test_optimized.py
```

#### 远程服务器运行

```bash
ssh jiuzhou@er074.utah.cloudlab.us
cd ~/Approximate

# 在tmux中运行（推荐）
tmux new -s scalability
python3 scalability_test_optimized.py

# 分离会话: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t scalability
```

**预期时间**：
- 原始版本：~2-4小时
- 优化版本：~15-30分钟（**快8-10倍**）

**输出**：
- `Result/scalability_fasttext_optimized/scalability_results_optimized.csv`
- 详细结果在 `Result/scalability_fasttext_optimized/` 目录下

### 2. Ablation Study（消融研究）

测试不同参数组合对MPAD性能的影响。

#### 运行单个数据集

```bash
# 本地
python ablation_study_optimized.py Fasttext

# 远程服务器
python3 ablation_study_optimized.py Fasttext
```

#### 运行所有数据集

```bash
# Fasttext (最快，推荐先测试)
python3 ablation_study_optimized.py Fasttext

# Isolet
python3 ablation_study_optimized.py Isolet

# PBMC3k
python3 ablation_study_optimized.py PBMC3k

# Arcene (最慢，维度最高)
python3 ablation_study_optimized.py Arcene
```

#### 使用多进程加速（仅在非常强的服务器上）

```bash
# 使用4个进程并行
python3 ablation_study_optimized.py Fasttext --processes 4
```

**⚠️ 注意**：由于MPAD内部已经使用了Numba并行，外部多进程可能导致资源竞争。推荐使用默认的单进程模式。

**预期时间（每个数据集）**：
- Fasttext: 原始~8小时 → 优化~1小时
- Isolet: 原始~12小时 → 优化~1.5小时
- PBMC3k: 原始~20小时 → 优化~2.5小时
- Arcene: 原始~40小时 → 优化~5小时

**输出**：
- `Result/ablation_{dataset}_optimized/ablation_results_optimized_{dataset}.csv`

### 3. 测试单个配置

如果只想测试特定的参数组合：

```python
# test_single_config.py
from main_program_optimized import main_evaluation_optimized

results, detail_file, summary_file = main_evaluation_optimized(
    dataset_name="Fasttext_test",
    train_file="training_vectors_01pct_Fasttext.npy",
    test_file="testing_vectors_01pct_Fasttext.npy",
    target_dim=128,
    b_percentage=1.0,
    alpha=0.1,
    k_values=[1, 10, 50],
    save_results=True,
    output_dir="Result/test"
)
```

然后运行：
```bash
python test_single_config.py
```

## 📊 结果文件

### Scalability Test 输出

```
Result/scalability_fasttext_optimized/
├── scalability_results_optimized.csv  # 汇总结果
├── Fasttext_01pct_detailed_results.csv
├── Fasttext_01pct_summary_report.txt
├── Fasttext_05pct_detailed_results.csv
├── Fasttext_05pct_summary_report.txt
├── Fasttext_10pct_detailed_results.csv
└── Fasttext_10pct_summary_report.txt
```

### Ablation Study 输出

```
Result/ablation_Fasttext_optimized/
├── ablation_results_optimized_Fasttext.csv  # 汇总结果
├── cache/  # 缓存的降维结果和ground truth
└── [各种详细结果文件]
```

## 🔍 验证优化效果

### 比较运行时间

```python
import pandas as pd

# 读取优化版结果
df_opt = pd.read_csv('Result/ablation_Fasttext_optimized/ablation_results_optimized_Fasttext.csv')

# 读取原始版结果（如果有）
df_orig = pd.read_csv('Result/ablation_Fasttext/ablation_results_Fasttext.csv')

# 比较MPAD运行时间
mpad_opt = df_opt[df_opt['method'] == 'MPAD']['dr_time'].mean()
mpad_orig = df_orig[df_orig['method'] == 'MPAD']['dr_time'].mean()

print(f"原始MPAD平均时间: {mpad_orig:.2f}s")
print(f"优化MPAD平均时间: {mpad_opt:.2f}s")
print(f"加速比: {mpad_orig/mpad_opt:.2f}x")
```

## ⚡ 性能预期

### 在28核Linux服务器上（er074.utah.cloudlab.us）

| 样本数 | 维度 | 原始MPAD | 优化MPAD | 加速比 |
|--------|------|----------|----------|--------|
| 1000   | 150  | ~90s     | ~8s      | 11x    |
| 2000   | 150  | ~210s    | ~16s     | 13x    |
| 4000   | 150  | ~490s    | ~37s     | 13x    |
| 8000   | 150  | ~1800s   | ~120s    | 15x    |

### 在本地Windows（已测试）

| 样本数 | 维度 | 原始MPAD | 优化MPAD | 加速比 |
|--------|------|----------|----------|--------|
| 1000   | 150  | 89.08s   | 17.03s   | 5.23x  |
| 2000   | 150  | 211.41s  | 16.49s   | 12.82x |
| 4000   | 150  | 488.50s  | 37.37s   | 13.07x |

## 🐛 故障排除

### 问题1：找不到数据文件

```bash
# 检查文件是否存在
ls -lh training_vectors_*

# 运行数据验证脚本
python3 quick_remote_check.py
```

### 问题2：Numba编译错误

```bash
# 重新安装Numba
pip install --upgrade numba

# 检查版本
python3 -c "import numba; print(numba.__version__)"
```

### 问题3：内存不足

```python
# 减小target_dim或使用更小的数据集
# 在ablation_study_optimized.py中修改配置
```

### 问题4：结果与原始版本不完全一致

这是正常的！优化版使用不同的计算顺序，L-BFGS-B优化器可能走不同路径。只要：
- 统计特性相似（均值、方差）
- Recall@k差异小于5%
- 整体趋势一致

就是可接受的。

## 📝 推荐的实验流程

### 阶段1：快速验证（1-2小时）

```bash
# 1. 测试优化效果
python3 test_optimized_mpad.py

# 2. 运行Fasttext scalability (最快)
python3 scalability_test_optimized.py

# 3. 运行Fasttext ablation (一个数据集)
python3 ablation_study_optimized.py Fasttext
```

### 阶段2：完整实验（1-2天）

```bash
# 在tmux中运行所有数据集的ablation study
tmux new -s ablation

python3 ablation_study_optimized.py Fasttext
python3 ablation_study_optimized.py Isolet
python3 ablation_study_optimized.py PBMC3k
python3 ablation_study_optimized.py Arcene

# 分离会话: Ctrl+B, D
```

### 阶段3：结果分析

```bash
# 下载结果到本地
scp -r jiuzhou@er074.utah.cloudlab.us:~/Approximate/Result ./

# 使用plot_scalability_results.py等脚本生成图表
```

## 💡 最佳实践

1. **Always use tmux** on remote servers to prevent interruption
2. **Monitor progress** with `tail -f` on log files
3. **Start with small datasets** (Fasttext) to verify setup
4. **Check disk space** before long runs
5. **Save intermediate results** frequently
6. **Document parameters** used in each run

## 📞 需要帮助？

如果遇到问题，请提供：
1. 运行的命令
2. 完整的错误信息
3. 系统信息（`uname -a`, CPU核心数）
4. 数据文件大小（`ls -lh training_*.npy`）

