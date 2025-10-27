# 远程服务器数据问题解决方案

## 问题诊断

您的远程服务器显示：
```
Loaded Fasttext 1%: (10, 300)
```

**这是错误的！** 应该显示约 `(7999, 300)` 或类似的数字。

## 原因

远程服务器上的 `training_vectors_01pct_Fasttext.npy` 文件**不完整或被截断**。

## 解决方案

### 步骤1：在本地验证数据文件

在本地Windows机器上（**不要SSH，在本地PowerShell执行**）：

```powershell
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD"

# 检查本地文件
python -c "import numpy as np; x = np.load('training_vectors_01pct_Fasttext.npy'); print(f'Shape: {x.shape}, Size: {x.nbytes/1024/1024:.2f} MB')"
```

**预期输出**：
```
Shape: (7999, 300), Size: 18.31 MB
```

如果本地文件也只有10个样本，那么需要重新生成数据。

### 步骤2：重新上传完整数据

#### 方法A：使用SCP（推荐）

```powershell
# 在本地PowerShell执行
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD"

scp training_vectors_01pct_Fasttext.npy jiuzhou@er074.utah.cloudlab.us:~/Approximate/
```

#### 方法B：使用WinSCP或FileZilla

1. 下载并安装 [WinSCP](https://winscp.net/) 或 [FileZilla](https://filezilla-project.org/)
2. 连接到 `er074.utah.cloudlab.us`，用户名 `jiuzhou`
3. 上传本地的 `training_vectors_01pct_Fasttext.npy` 到远程的 `~/Approximate/`

#### 方法C：使用rsync（如果安装了WSL或Cygwin）

```bash
rsync -avz --progress training_vectors_01pct_Fasttext.npy jiuzhou@er074.utah.cloudlab.us:~/Approximate/
```

### 步骤3：在远程服务器验证

SSH到远程服务器：

```bash
ssh jiuzhou@er074.utah.cloudlab.us
cd ~/Approximate
```

运行验证脚本：

```bash
python3 quick_remote_check.py
```

**预期输出**：
```
======================================================================
DATA FILE VERIFICATION
======================================================================

[INFO] File found: training_vectors_01pct_Fasttext.npy
  File size: 19,198,400 bytes (18.31 MB)
  Data shape: (7999, 300)
  Data type: float64
  Memory usage: 18.31 MB

[OK] Data looks good!
  7999 samples × 300 features
  Mean: 0.000123, Std: 0.123456
  Min: -0.987654, Max: 0.876543

[READY] You can now run: python3 test_optimized_mpad.py
======================================================================
```

### 步骤4：重新运行测试

```bash
# 确保环境变量已设置
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_THREADING_LAYER=omp
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 运行测试
python3 test_optimized_mpad.py
```

## 如果本地数据也只有10个样本

这意味着数据预处理脚本有问题。需要重新生成数据：

```bash
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD\Approximate"

# 运行数据预处理脚本
python data_preprocessing.py
```

然后检查生成的文件：

```python
python -c "import numpy as np; print('Training:', np.load('training_vectors_01pct_Fasttext.npy').shape)"
```

## 替代方案：使用其他数据集

如果Fasttext数据有问题，可以先用其他数据集测试：

### 测试Isolet数据集

```bash
# 在远程服务器上
cd ~/Approximate

# 从本地上传Isolet数据（如果还没上传）
# scp training_vectors_Isolet.npy jiuzhou@er074.utah.cloudlab.us:~/Approximate/
```

修改 `test_optimized_mpad.py` 第115行左右：

```python
# 原来的
data_file = "training_vectors_01pct_Fasttext.npy"

# 改为
data_file = "training_vectors_Isolet.npy"
```

Isolet数据集有约6500个样本，617维特征，足够测试性能。

## 快速测试脚本

如果你只想快速验证优化效果，可以创建一个小型测试：

```python
# test_small.py
import numpy as np
import time
from main_program import MPAD as BaselineMPAD
from mpad_optimized import MPAD_Optimized

# 生成随机数据
np.random.seed(42)
X = np.random.randn(2000, 300)

print("Testing with synthetic data (2000 samples, 300 dims)")

# Baseline
print("\n[Baseline MPAD]")
np.random.seed(42)
t0 = time.time()
mpad_b = BaselineMPAD(b_percentage=1.0, alpha=0.1, target_dim=50)
X_b = mpad_b.fit_transform(X.copy())
dt_b = time.time() - t0
print(f"  Time: {dt_b:.2f}s")

# Optimized
print("\n[Optimized MPAD]")
np.random.seed(42)
t0 = time.time()
mpad_o = MPAD_Optimized(b_percentage=1.0, alpha=0.1, target_dim=50)
X_o = mpad_o.fit_transform(X.copy())
dt_o = time.time() - t0
print(f"  Time: {dt_o:.2f}s")

print(f"\n[Speedup] {dt_b/dt_o:.2f}x")
```

运行：
```bash
python3 test_small.py
```

## 总结

**最可能的问题**：上传过程中文件被截断或传输不完整。

**解决方法**：
1. 验证本地文件完整性
2. 重新上传完整文件
3. 在远程验证文件
4. 重新运行测试

**检查清单**：
- [ ] 本地文件有~8000个样本
- [ ] 上传命令成功执行（无错误）
- [ ] 远程文件大小约18MB
- [ ] 远程文件有~8000个样本
- [ ] 环境变量已设置
- [ ] 测试脚本运行成功

如果仍有问题，请提供：
1. 本地文件的shape
2. 上传命令的输出
3. 远程文件大小（`ls -lh training_vectors_01pct_Fasttext.npy`）

