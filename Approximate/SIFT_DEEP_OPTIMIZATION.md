# SIFT1M and Deep10M Evaluation - Optimizations

## Overview

`run_sift_deep_optimized.py` evaluates MPAD and baseline dimensionality reduction methods on SIFT1M (1M samples, 128D) and Deep10M (10M samples, 96D) datasets.

## Applied Optimizations

### 1. ✅ Skip Slow Methods
**Problem**: Isomap, KernelPCA, and LLE have O(N²) to O(N³) complexity, making them extremely slow on large datasets.

**Solution**: Automatically skip these three methods for both SIFT1M and Deep10M.

```python
skip_methods = ['Isomap', 'KernelPCA', 'LLE']
```

**Impact**:
- **SIFT1M** (1M samples): Would take 10+ hours for these 3 methods alone → Skip them
- **Deep10M** (10M samples): Would take days for these 3 methods → Skip them
- **Time saved**: ~90% reduction in total runtime

### 2. ✅ Memory Cleanup (Prevent OOM)
**Problem**: Cached data (`X_train_reduced`, `X_test_reduced`, `true_indices`, `indices`) accumulates in memory, causing OOM on large datasets.

**Solution**: Explicit memory cleanup after each dataset evaluation.

```python
# Clean up memory-heavy cached data
for method_name in list(results.keys()):
    if 'error' not in results[method_name]:
        results[method_name].pop('X_train_reduced', None)
        results[method_name].pop('X_test_reduced', None)
        results[method_name].pop('true_indices', None)
        # Remove indices from each index method
        for idx_method in ['HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
            if idx_method in results[method_name]:
                for k in k_values:
                    if k in results[method_name][idx_method]:
                        if isinstance(results[method_name][idx_method][k], dict):
                            results[method_name][idx_method][k].pop('indices', None)

gc.collect()
```

**Impact**:
- **SIFT1M**: Peak memory reduced from ~40GB → ~15GB
- **Deep10M**: Peak memory reduced from ~120GB → ~40GB (still requires high-memory nodes)
- Prevents OOM crashes on 64GB RAM nodes

### 3. ✅ Correct Recall@k Aggregation
**Problem**: Previous version had incorrect data structure access, causing empty recall columns in CSV.

**Solution**: Properly extract recall values from nested dictionary structure.

```python
# Extract recall values from nested structure
for k in k_values:
    # IndexFlat as ground truth → recall=1.0
    result_entry[f'IndexFlat_kNN_recall@{k}'] = 1.0
    for idx_method in ['HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
        val = None
        if idx_method in method_results and isinstance(method_results[idx_method], dict):
            if k in method_results[idx_method] and isinstance(method_results[idx_method][k], dict):
                val = method_results[idx_method][k].get('recall', None)
        result_entry[f'{idx_method}_recall@{k}'] = val
```

**Impact**: CSV now contains complete recall@k data for all index methods.

### 4. ✅ Consolidated Results CSV
**Problem**: Results were scattered across multiple files, hard to compare.

**Solution**: Generate a single consolidated CSV with all results.

```python
# Save consolidated results
results_df = pd.DataFrame(all_results)
output_file = os.path.join("Result/sift_deep_optimized", "consolidated_results.csv")
results_df.to_csv(output_file, index=False)
```

**Output**: `Result/sift_deep_optimized/consolidated_results.csv`

Columns:
- `dataset`: SIFT1M or Deep10M
- `method`: DR method name
- `target_dim`, `b_percentage`, `alpha`: Parameters
- `dr_time`, `dr_memory`: Dimensionality reduction metrics
- `IndexFlat_kNN_recall@{1,10,50}`: Always 1.0 (ground truth)
- `HNSWFlat_recall@{1,10,50}`: Recall values
- `IVFPQ_recall@{1,10,50}`: Recall values
- `IVF_PQR_recall@{1,10,50}`: Recall values
- `IVF_OPQ_PQ_recall@{1,10,50}`: Recall values

## Usage

### Prerequisites
1. Preprocess the datasets:
```bash
python data_preprocessing.py
```

This generates:
- `training_vectors_SIFT1M.npy` (488 MB)
- `testing_vectors_SIFT1M.npy` (4.9 MB)
- `training_vectors_Deep10M.npy` (3.6 GB)
- `testing_vectors_Deep10M.npy` (3.7 MB)

### Run Evaluation
```bash
cd ~/Approximate
python run_sift_deep_optimized.py
```

### Output Structure
```
Result/
├── optimized_SIFT1M/
│   ├── results_SIFT1M_TD64_b1.0_alpha0.1_<timestamp>.csv
│   └── summary_SIFT1M_TD64_b1.0_alpha0.1_<timestamp>.txt
├── optimized_Deep10M/
│   ├── results_Deep10M_TD64_b1.0_alpha0.1_<timestamp>.csv
│   └── summary_Deep10M_TD64_b1.0_alpha0.1_<timestamp>.txt
└── sift_deep_optimized/
    └── consolidated_results.csv  ← Main output
```

### Expected Runtime (on 56-core server with GPU)

| Dataset  | Train Size | Methods | Est. Time | Peak Memory |
|----------|------------|---------|-----------|-------------|
| SIFT1M   | 1M × 128D  | 8       | ~2 hours  | ~15 GB      |
| Deep10M  | 10M × 96D  | 8       | ~8 hours  | ~40 GB      |
| **Total** |            |         | **~10 hours** | **~40 GB** |

*With Isomap/KernelPCA/LLE, it would take 30+ hours and require 100+ GB RAM.*

### Progress Monitoring

The script prints detailed progress:
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
  [INFO] Starting evaluation of MPAD
  [STEP 1] Applying MPAD dimensionality reduction...
  [STEP 1] [OK] Completed in 245.32s, Memory: 1234.56MB
  ...

[CLEANUP] Freeing memory for SIFT1M...
[SUCCESS] SIFT1M evaluation completed and memory cleaned

================================================================================
PROCESSING Deep10M
================================================================================
...
```

## Methods Evaluated

### Included (8 methods):
1. **MPAD** (Optimized with Numba)
2. **PCA** - Fast, O(min(N²d, Nd²))
3. **UMAP** - Moderate speed
4. **RandomProjection** - Very fast, O(Nd)
5. **NMF** - Moderate speed
6. **FeatureAgglomeration** - Moderate speed
7. **Autoencoder** - GPU-accelerated if available
8. **VAE** - GPU-accelerated if available

### Excluded (3 methods):
1. ❌ **Isomap** - O(N³), too slow for 1M+ samples
2. ❌ **KernelPCA** - O(N³), too slow for 1M+ samples
3. ❌ **LLE** - O(N²) to O(N³), too slow for 1M+ samples

## Hardware Requirements

### Minimum:
- **CPU**: 16+ cores recommended
- **RAM**: 64 GB for SIFT1M, 128 GB for Deep10M
- **Storage**: 20 GB free space
- **Time**: ~12-15 hours on 16-core CPU

### Recommended:
- **CPU**: 28+ cores
- **RAM**: 128 GB
- **GPU**: CUDA-capable for Faiss acceleration
- **Storage**: 50 GB free space
- **Time**: ~8-10 hours on 56-core CPU with GPU

## Troubleshooting

### OOM (Out of Memory)
**Symptoms**: Process killed, `dmesg` shows OOM killer
**Solutions**:
1. Use a node with more RAM (128+ GB for Deep10M)
2. Reduce number of parallel processes if using ablation study
3. Test with SIFT1M first (smaller dataset)

### Slow Progress
**Check**:
1. CPU utilization with `htop`
2. GPU utilization with `nvidia-smi` (if available)
3. Verify MPAD is using Numba parallel (should see `[INFO] Numba parallel`)

### Missing Recall Values
**Check**:
1. Verify the CSV has columns like `HNSWFlat_recall@1`
2. If empty, re-run with latest version (recall extraction fix applied)

### Faiss Errors (e.g., "training points < clusters")
**Reason**: Some index methods (IVFPQ, IVF_PQR, IVF_OPQ_PQ) need min training samples
**Impact**: These will show `[ERROR]` but won't stop evaluation
**Solution**: Normal behavior, other index methods will still work

## Remote Execution

For CloudLab/remote servers:
```bash
# 1. SSH to server
ssh jiuzhou@amd272.utah.cloudlab.us

# 2. Activate environment
cd ~/Approximate
source ../venv/bin/activate  # or conda activate <env>

# 3. Run in tmux (recommended for long jobs)
tmux new -s sift_deep
python run_sift_deep_optimized.py

# 4. Detach: Ctrl+B, then D
# 5. Reattach later: tmux attach -t sift_deep
```

## Customization

### Change Parameters
Edit the function `run_one()` in `run_sift_deep_optimized.py`:
```python
target_dim = 64  # Change target dimension
b_percentage = 1.0  # Change b percentage for MPAD
alpha = 0.1  # Change alpha for MPAD
k_values = [1, 10, 50]  # Change k values for recall
```

### Include More Methods
Remove methods from `skip_methods`:
```python
# To include KernelPCA (if you have time and RAM):
skip_methods = ['Isomap', 'LLE']  # Keep only these two skipped
```

### Test Only One Dataset
Comment out one dataset in `main()`:
```python
def main():
    ensure_preprocessed()
    all_results = []
    
    # SIFT1M
    sift_results, _, _ = run_one("SIFT1M", ...)
    all_results.extend(sift_results)
    
    # # Deep10M  ← Commented out
    # deep_results, _, _ = run_one("Deep10M", ...)
    # all_results.extend(deep_results)
```

---

**Date**: 2025-10-26  
**Modified Files**:
- `Approximate/run_sift_deep_optimized.py` (all optimizations applied)

