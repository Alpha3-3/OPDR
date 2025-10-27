# Recall Data Extraction Fix

## Date
October 27, 2025

## Problem

After implementing the ground truth fix (computing ground truth from original space instead of reduced space), the scalability test was generating empty recall columns and crashing with:

```
TypeError: argument of type 'float' is not iterable
```

## Root Causes

### Issue 1: Type Error in `save_summary_report`

**Location:** `main_program.py:1026`

**Problem:**
The function was iterating over `method_results.items()` but not properly handling new scalar keys added for ground truth tracking:
- `gt_time_orig` (float)
- `gt_memory_orig` (float)

When the code tried to check `if k in index_results:` on these float values, Python raised `TypeError`.

**Fix:**
```python
# OLD (incomplete skip list)
if index_name in ['dr_time', 'dr_memory', 'gt_time', 'gt_memory']:
    continue

# NEW (comprehensive skip list + type checking)
if index_name in ['dr_time', 'dr_memory', 'gt_time', 'gt_memory', 
                'gt_time_orig', 'gt_memory_orig',
                'X_train_reduced', 'X_test_reduced', 'true_indices']:
    continue

# Skip if not a dictionary (should be dict mapping k -> results)
if not isinstance(index_results, dict):
    continue
```

### Issue 2: Incorrect Recall Value for IndexFlat_kNN

**Location:** All experiment scripts
- `scalability_test_optimized.py:111`
- `ablation_study_optimized.py:272`
- `run_sift_deep_optimized.py:96`

**Problem:**
After the ground truth fix, these scripts were still hardcoding:
```python
result_entry[f'IndexFlat_kNN_recall@{k}'] = 1.0
```

This was **fundamentally wrong** because:
- Ground truth is now from the **original high-dimensional space**
- IndexFlat_kNN on **reduced space** should have recall < 1.0
- Hardcoding 1.0 ignored the actual computed recall values

**Fix:**
```python
# OLD (incorrect assumption)
for k in k_values:
    result_entry[f'IndexFlat_kNN_recall@{k}'] = 1.0
    for idx_method in ['HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
        # extract recall...

# NEW (correct extraction from all index methods)
for k in k_values:
    for idx_method in ['IndexFlat_kNN', 'HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
        val = None
        if idx_method in method_results and isinstance(method_results[idx_method], dict):
            if k in method_results[idx_method] and isinstance(method_results[idx_method][k], dict):
                val = method_results[idx_method][k].get('recall', None)
        result_entry[f'{idx_method}_recall@{k}'] = val
```

### Issue 3: Obsolete Helper Function

**Location:** `scalability_test_optimized.py:22`

**Problem:**
The function `idx_method_reconstruct()` was used to rename index methods but is no longer needed since we now use consistent naming.

**Fix:**
Removed the function entirely.

## Files Modified

1. ✅ `main_program.py`
   - Enhanced skip list in `save_summary_report`
   - Added type checking for `index_results`

2. ✅ `scalability_test_optimized.py`
   - Fixed recall extraction to include IndexFlat_kNN
   - Removed hardcoded `IndexFlat_kNN_recall@k = 1.0`
   - Deleted `idx_method_reconstruct()` function

3. ✅ `ablation_study_optimized.py`
   - Fixed recall extraction to include IndexFlat_kNN
   - Removed hardcoded `IndexFlat_kNN_recall@k = 1.0`

4. ✅ `run_sift_deep_optimized.py`
   - Fixed recall extraction to include IndexFlat_kNN
   - Removed hardcoded `IndexFlat_kNN_recall@k = 1.0`

## Expected Results After Fix

### CSV Output Structure

Each result row should now contain:

```
method,target_dim,b_percentage,alpha,dr_time,dr_memory,
IndexFlat_kNN_recall@1,IndexFlat_kNN_recall@10,IndexFlat_kNN_recall@50,
HNSWFlat_recall@1,HNSWFlat_recall@10,HNSWFlat_recall@50,
IVFPQ_recall@1,IVFPQ_recall@10,IVFPQ_recall@50,
IVF_PQR_recall@1,IVF_PQR_recall@10,IVF_PQR_recall@50,
IVF_OPQ_PQ_recall@1,IVF_OPQ_PQ_recall@10,IVF_OPQ_PQ_recall@50
```

### Recall Value Interpretation

Now all recall values correctly represent:

**Recall@k = |kNN_reduced ∩ kNN_original| / k**

Where:
- `kNN_original` = k nearest neighbors in **original high-dimensional space** (ground truth)
- `kNN_reduced` = k nearest neighbors in **reduced space** using specific index method

**Expected typical values:**
- **IndexFlat_kNN:** 0.7-0.95 (exact search on reduced space, quality depends on DR method)
- **HNSWFlat:** 0.6-0.9 (approximate search with high recall)
- **IVFPQ:** 0.4-0.8 (faster but lower recall)
- **IVF_PQR:** 0.4-0.75 (similar to IVFPQ)
- **IVF_OPQ_PQ:** 0.5-0.85 (optimized quantization, better than IVFPQ)

The exact values will vary by:
- Dataset characteristics
- Dimensionality reduction quality
- Target dimension
- k value (typically recall@50 > recall@10 > recall@1)

## Testing

To verify the fix works:

```bash
# On remote server
cd ~/Approximate
source ~/venv_approximate/bin/activate

# Quick test on small dataset
python -c "
from scalability_test_optimized import run_fasttext_scalability_test_optimized
import pandas as pd

# Run test
run_fasttext_scalability_test_optimized()

# Check results
df = pd.read_csv('Result/scalability_fasttext_optimized/scalability_results.csv')
print('Recall columns:')
for col in df.columns:
    if 'recall' in col:
        print(f'  {col}: {df[col].notna().sum()}/{len(df)} non-null values')
        if df[col].notna().sum() > 0:
            print(f'    Range: [{df[col].min():.3f}, {df[col].max():.3f}]')
"
```

Expected output:
```
Recall columns:
  IndexFlat_kNN_recall@1: 33/33 non-null values
    Range: [0.750, 0.950]
  IndexFlat_kNN_recall@10: 33/33 non-null values
    Range: [0.800, 0.980]
  IndexFlat_kNN_recall@50: 33/33 non-null values
    Range: [0.850, 0.995]
  HNSWFlat_recall@1: 33/33 non-null values
    Range: [0.650, 0.900]
  ...
```

## Impact on Previous Results

**IMPORTANT:** All previous experimental results with empty recall columns or `IndexFlat_kNN_recall@k = 1.0` are **invalid** and must be re-run.

This includes:
- ❌ Any scalability test results before this fix
- ❌ Any ablation study results before this fix
- ❌ Any SIFT1M/Deep10M/Fasttext 100% results before this fix

## Related Fixes

This fix builds on the previous ground truth correction documented in:
- `GROUND_TRUTH_FIX.md` - Changed ground truth computation from reduced space to original space

Together, these two fixes ensure:
1. Ground truth is correct (from original space)
2. Recall extraction is correct (all index methods, including IndexFlat_kNN)
3. No crashes from type errors on scalar values

## Next Steps

1. ✅ Fix applied to all experiment scripts
2. 🔄 Re-run experiments on remote server with correct recall extraction
3. 📊 Validate results show expected recall ranges (not 1.0 for all, not empty)
4. 📈 Generate plots with corrected recall data

---

**Date Fixed:** October 27, 2025  
**Tested By:** [Pending]  
**Status:** Ready for re-execution

