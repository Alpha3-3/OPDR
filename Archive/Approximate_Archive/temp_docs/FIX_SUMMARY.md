# Fix Summary: Recall Data Extraction Issue

## Date
**October 27, 2025**

## Issue Report

User reported two critical issues after running experiments on remote server:

1. **Empty recall data** in all CSV output files
2. **Type error crash**: `TypeError: argument of type 'float' is not iterable`

## Root Causes Identified

### 1. Type Error in Summary Report Generation

**File:** `main_program.py` line 1026

**Problem:**
- New keys added for ground truth tracking: `gt_time_orig`, `gt_memory_orig`
- These are scalar floats, not dictionaries
- Code attempted to check `if k in index_results` on floats → TypeError

**Solution:**
- Extended skip list to include new scalar keys
- Added type check: `if not isinstance(index_results, dict): continue`

### 2. Hardcoded IndexFlat Recall Values

**Files:**
- `scalability_test_optimized.py` line 111
- `ablation_study_optimized.py` line 272
- `run_sift_deep_optimized.py` line 96

**Problem:**
- After ground truth fix, these scripts hardcoded: `IndexFlat_kNN_recall@k = 1.0`
- This is **incorrect** because ground truth is now from original space
- IndexFlat on reduced space should have recall < 1.0
- Hardcoding 1.0 overwrote actual computed recall values

**Solution:**
- Removed hardcoded assignments
- Changed loop to iterate over ALL index methods including `IndexFlat_kNN`
- Extract recall from actual results: `method_results[idx_method][k].get('recall')`

## Changes Made

### Modified Files

| File | Changes |
|------|---------|
| `main_program.py` | Extended skip list + type checking in `save_summary_report` |
| `scalability_test_optimized.py` | Fixed recall extraction, removed hardcoded 1.0, deleted obsolete helper |
| `ablation_study_optimized.py` | Fixed recall extraction, removed hardcoded 1.0 |
| `run_sift_deep_optimized.py` | Fixed recall extraction, removed hardcoded 1.0 |

### Code Changes (Before → After)

**Before (Incorrect):**
```python
# Hardcoded IndexFlat recall
for k in k_values:
    result_entry[f'IndexFlat_kNN_recall@{k}'] = 1.0
    for idx_method in ['HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
        # extract from method_results...
```

**After (Correct):**
```python
# Extract all index methods including IndexFlat
for k in k_values:
    for idx_method in ['IndexFlat_kNN', 'HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']:
        val = None
        if idx_method in method_results and isinstance(method_results[idx_method], dict):
            if k in method_results[idx_method] and isinstance(method_results[idx_method][k], dict):
                val = method_results[idx_method][k].get('recall', None)
        result_entry[f'{idx_method}_recall@{k}'] = val
```

## Expected Results After Fix

### CSV Output

All recall columns should now be populated:

```csv
method,target_dim,b_percentage,alpha,dr_time,dr_memory,
IndexFlat_kNN_recall@1,IndexFlat_kNN_recall@10,IndexFlat_kNN_recall@50,
HNSWFlat_recall@1,HNSWFlat_recall@10,HNSWFlat_recall@50,
IVFPQ_recall@1,IVFPQ_recall@10,IVFPQ_recall@50,
IVF_PQR_recall@1,IVF_PQR_recall@10,IVF_PQR_recall@50,
IVF_OPQ_PQ_recall@1,IVF_OPQ_PQ_recall@10,IVF_OPQ_PQ_recall@50
PCA,128,1.0,0.1,2.34,150.5,0.85,0.92,0.95,0.78,0.88,0.91,...
```

### Typical Recall Ranges

With ground truth from original space:

| Index Method | Expected Recall@10 |
|--------------|-------------------|
| IndexFlat_kNN | 0.75 - 0.95 |
| HNSWFlat | 0.65 - 0.90 |
| IVFPQ | 0.45 - 0.80 |
| IVF_PQR | 0.40 - 0.75 |
| IVF_OPQ_PQ | 0.50 - 0.85 |

**Key insight:** IndexFlat recall < 1.0 because:
- Ground truth = exact kNN in **original** high-D space
- IndexFlat recall = exact kNN in **reduced** space vs ground truth
- Dimensionality reduction loses some neighborhood information

## Testing

### Local Test

```bash
cd Approximate
python test_recall_fix.py
```

Expected output:
```
[CHECK] Found 10 recall columns:
  [OK] IndexFlat_kNN_recall@1    :  2/ 2 non-null  Range: [0.750, 0.850]
  [OK] IndexFlat_kNN_recall@10   :  2/ 2 non-null  Range: [0.800, 0.900]
  [OK] HNSWFlat_recall@1         :  2/ 2 non-null  Range: [0.650, 0.750]
  ...

[CHECK] IndexFlat_kNN columns: 2
  [OK] IndexFlat_kNN_recall@1 mean = 0.800 (correct, < 1.0)
  [OK] IndexFlat_kNN_recall@10 mean = 0.850 (correct, < 1.0)

[SUCCESS] All checks passed!
```

### Remote Deployment

```bash
# Upload fixed files
cd Approximate
bash upload_recall_fix.sh

# On remote server
ssh jiuzhou@er074.utah.cloudlab.us
cd ~/Approximate
source ~/venv_approximate/bin/activate
tmux new -s rerun
python rerun_all_experiments.py --all
```

## Impact Assessment

### Invalid Previous Results

❌ All results generated **before this fix** with:
- Empty recall columns
- `IndexFlat_kNN_recall@k = 1.0`
- Type error crashes

Must be **discarded and re-run**.

### Valid Results

✅ Results generated **after this fix** will have:
- All recall columns populated with actual computed values
- IndexFlat recall correctly reflecting DR quality (< 1.0)
- No type errors or crashes

## Related Documentation

This fix builds on:
- **`GROUND_TRUTH_FIX.md`** - Changed ground truth from reduced to original space
- **`MEMORY_AND_PARALLELIZATION_FIX.md`** - Fixed OOM and CPU utilization

Together these ensure:
1. ✅ Correct ground truth (from original space)
2. ✅ Correct recall extraction (all methods, no hardcoding)
3. ✅ No type errors on scalar values
4. ✅ Efficient memory usage and parallelization

## Timeline

| Date | Event |
|------|-------|
| Oct 26 | Ground truth fix implemented |
| Oct 27 | User reports empty recall data + type error |
| Oct 27 | Root cause identified: hardcoded 1.0 + type error |
| Oct 27 | Fix implemented and tested |
| Oct 27 | Ready for re-execution on remote server |

## Verification Checklist

Before accepting results as valid:

- [ ] No empty recall columns in CSV
- [ ] No `TypeError: argument of type 'float' is not iterable`
- [ ] IndexFlat_kNN recall values are < 1.0 (not 1.0)
- [ ] IndexFlat_kNN recall shows reasonable values (0.7-0.95 range)
- [ ] All 5 index methods have recall data for all k values
- [ ] Recall@50 > Recall@10 > Recall@1 (generally expected)

## Next Steps

1. ✅ Upload fixed scripts to remote server
2. 🔄 Re-run all experiments: `python rerun_all_experiments.py --all`
3. 📊 Verify recall columns are populated correctly
4. 📈 Generate plots with corrected data
5. 📝 Update paper with valid experimental results

---

**Status:** Fixed and Ready for Re-execution  
**Priority:** Critical (blocks all experimental results)  
**Estimated Re-run Time:** ~6-8 hours for full pipeline

