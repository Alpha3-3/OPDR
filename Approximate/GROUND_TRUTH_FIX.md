# CRITICAL FIX: Ground Truth from Original Space

## 🚨 Major Issue Discovered and Fixed

### ❌ Previous INCORRECT Implementation

**What was wrong:**
```python
# Step 1: Apply DR
X_train_reduced, X_test_reduced = method_func(X_train, X_test, target_dim)

# Step 2: Calculate ground truth on REDUCED space ❌ WRONG!
true_indices = IndexFlat_kNN(X_train_reduced, X_test_reduced, k)

# Step 3: Compare other index methods (on reduced space) vs ground truth (also on reduced space)
recall = compare(other_index_results, true_indices)
```

**The problem:**
- Ground truth was computed on **reduced-dimensional space**
- This only tested **index method accuracy**, NOT **dimensionality reduction quality**!
- All DR methods would show similar high recall because they're comparing against themselves

### ✅ Correct Implementation (NOW FIXED)

**What it should be:**
```python
# Step 0: Calculate ground truth on ORIGINAL space (ONCE for all methods)
true_indices_orig = IndexFlat_kNN(X_train_ORIGINAL, X_test_ORIGINAL, k) ✅ CORRECT!

# For each DR method:
  # Step 1: Apply DR
  X_train_reduced, X_test_reduced = method_func(X_train, X_test, target_dim)
  
  # Step 2: Test ALL index methods (including exact kNN) on REDUCED space
  for each index_method:
      pred_indices = index_method(X_train_reduced, X_test_reduced, k)
      
      # Step 3: Compare against ORIGINAL-space ground truth ✅ CORRECT!
      recall = compare(pred_indices, true_indices_orig)
```

**Why this is correct:**
- Ground truth is computed on **original high-dimensional space**
- Each DR method + index method combination is tested on **reduced space**
- Recall measures: "How many true neighbors (from original space) are preserved after DR?"
- This correctly evaluates **dimensionality reduction quality** + **index method accuracy**

---

## 📊 What This Means

### Impact on Recall@k Values

#### Before Fix (WRONG):
- **IndexFlat on reduced space**: Recall ≈ 1.0 (always, by definition)
- **HNSWFlat on reduced space**: Recall ≈ 0.95-0.99 (close to 1.0)
- **IVFPQ on reduced space**: Recall ≈ 0.85-0.95
- **Result**: Only shows index approximation error, NOT DR quality

#### After Fix (CORRECT):
- **IndexFlat on reduced space**: Recall ≈ 0.70-0.95 (depends on DR quality!)
- **HNSWFlat on reduced space**: Recall ≈ 0.65-0.92
- **IVFPQ on reduced space**: Recall ≈ 0.55-0.85
- **Result**: Shows BOTH DR quality AND index approximation error

### Example Interpretation

**MPAD with TD=64 on Fasttext (300D → 64D):**
- `IndexFlat_kNN_recall@10 = 0.85`: After reducing from 300D to 64D, exact kNN finds 85% of the original true neighbors
- `HNSWFlat_recall@10 = 0.82`: HNSW approximate kNN finds 82% (3% approximation error + 15% DR error)
- `IVFPQ_recall@10 = 0.75`: IVFPQ finds 75% (10% approximation error + 15% DR error)

---

## 🔧 What Was Changed

### Modified Files

#### 1. `Approximate/main_program.py`

**Changes to `evaluate_method()`:**
- Added parameter: `true_indices_orig` (ground truth from original space)
- Removed: Calculating ground truth on reduced space
- Now uses: Pre-computed ground truth from original space

```python
def evaluate_method(method_name, method_func, X_train, X_test, target_dim, 
                    index_methods, k_values, true_indices_orig=None):  # NEW PARAMETER
    """
    Args:
        true_indices_orig: Ground truth indices from ORIGINAL space (required)
    """
    # ... apply DR ...
    
    # Use ground truth from ORIGINAL space (passed in)
    true_indices = true_indices_orig  # ✅ CORRECT
    
    # Test all index methods on REDUCED space
    for index_method in index_methods:
        pred_indices = index_method(X_train_reduced, X_test_reduced, k)
        recall = compare(pred_indices, true_indices)  # Compare vs ORIGINAL-space GT
```

**Changes to `main_evaluation()`:**
- Added: Compute ground truth on **ORIGINAL space** (once, before all methods)
- Modified: Pass `true_indices_orig` to each `evaluate_method()` call

```python
def main_evaluation(...):
    # Load original data
    X_train = np.load(train_file)
    X_test = np.load(test_file)
    
    # ===== COMPUTE GROUND TRUTH ON ORIGINAL SPACE (ONCE) =====
    print("COMPUTING GROUND TRUTH IN ORIGINAL SPACE")
    true_indices_orig = IndexFlat_kNN(X_train, X_test, max(k_values))  # ✅ ORIGINAL SPACE
    
    # Evaluate each DR method
    for method_name, method_func in methods.items():
        results = evaluate_method(
            method_name, method_func, X_train, X_test, target_dim,
            index_methods, k_values, 
            true_indices_orig=true_indices_orig  # ✅ PASS ORIGINAL-SPACE GT
        )
```

#### 2. `Approximate/main_program_optimized.py`

- Same changes as `main_program.py`
- Ensures optimized MPAD uses correct ground truth evaluation

#### 3. All scripts using these functions:
- `scalability_test_optimized.py` ✅ Automatically fixed (uses `main_program_optimized`)
- `ablation_study_optimized.py` ✅ Automatically fixed (uses `main_program_optimized`)
- `run_sift_deep_optimized.py` ✅ Automatically fixed (uses `main_program_optimized`)

---

## 🎯 Expected Behavior After Fix

### Output Changes

#### NEW: Ground Truth Computation Message
```
================================================================================
COMPUTING GROUND TRUTH IN ORIGINAL SPACE
================================================================================
[INFO] Computing exact kNN on ORIGINAL data (train=(10000, 300), test=(2000, 300))
[INFO] This ground truth will be used to evaluate ALL dimensionality reduction methods
[OK] Ground truth computed in 2.3456s, Memory: 12.34MB
[INFO] Ground truth shape: (2000, 50)
================================================================================
```

#### Per-Method Evaluation:
```
================================================================================
Evaluating: MPAD
================================================================================
  [STEP 1] Applying MPAD dimensionality reduction...
  [STEP 1] [OK] Completed in 15.23s, Memory: 123.45MB
  [STEP 2] Using ground truth from ORIGINAL space (pre-computed)  ← NEW MESSAGE
  [STEP 3] Evaluating IndexFlat_kNN on reduced space...
    [OK] k=10: Recall=0.8523, Time=0.15s  ← LOWER recall (correct!)
  [STEP 3] Evaluating HNSWFlat on reduced space...
    [OK] k=10: Recall=0.8245, Time=0.12s
```

### Recall@k Value Ranges (Expected)

| Method | TD=64 | TD=128 | TD=256 |
|--------|-------|--------|--------|
| **MPAD** | 0.75-0.90 | 0.85-0.95 | 0.92-0.98 |
| **PCA** | 0.60-0.80 | 0.75-0.88 | 0.85-0.93 |
| **UMAP** | 0.65-0.85 | 0.78-0.90 | 0.87-0.94 |
| **Random Proj** | 0.50-0.70 | 0.65-0.80 | 0.75-0.88 |

*These are rough estimates for Fasttext with k=10. Actual values depend on dataset.*

---

## ⚠️ Important Notes

### 1. Performance Impact
- **Original-space GT computation**: Adds one-time cost at the start
  - SIFT1M (1M × 128D): ~30-60 seconds
  - Deep10M (10M × 96D): ~5-10 minutes
  - Fasttext 10% (8K × 300D): ~1-2 seconds
- **Benefit**: This cost is paid ONCE, not per-method
- **Net impact**: Slightly longer total runtime, but same per-method time

### 2. Memory Impact
- Ground truth indices are stored in memory (M × k, dtype=int64)
- Example: 10K test samples, k=50 → 10K × 50 × 8 bytes = 4 MB (negligible)
- No significant memory increase

### 3. Backward Compatibility
- **CSV format**: Same columns, same structure
- **Recall values**: Will be LOWER (correct values, not inflated)
- **Comparison**: Cannot compare results before/after fix (different metrics)

### 4. Re-running Experiments
- **All previous results are INVALID** (compared against wrong ground truth)
- **Must re-run**:
  - All scalability tests
  - All ablation studies
  - SIFT1M and Deep10M evaluations
- **Re-run is ESSENTIAL** to get correct DR quality evaluation

---

## 📚 Theoretical Background

### Why Original-Space Ground Truth?

**Goal of Dimensionality Reduction Evaluation:**
- Measure how well DR preserves the **neighborhood structure** of the original space
- If point A's neighbors in original space are {B, C, D}, do they remain neighbors after DR?

**Correct Evaluation Process:**
1. Find true neighbors in **original high-dimensional space**
2. Apply dimensionality reduction
3. Find neighbors in **reduced space** (using any index method)
4. Compare: How many original neighbors are still found in reduced space?

**This measures:**
- **DR quality**: How much information is preserved?
- **Index accuracy**: How well does the index approximate exact kNN?

---

## ✅ Verification

### How to Verify the Fix Works

#### Test Script (Quick Check):
```python
import numpy as np
from main_program_optimized import main_evaluation_optimized

# Small test
X_train = np.random.randn(100, 50)
X_test = np.random.randn(20, 50)
X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

np.save("test_train.npy", X_train)
np.save("test_test.npy", X_test)

results, _, _ = main_evaluation_optimized(
    "test", "test_train.npy", "test_test.npy",
    target_dim=10, b_percentage=1.0, alpha=0.1, k_values=[5],
    save_results=False
)

# Check: IndexFlat recall should be < 1.0 (e.g., 0.6-0.9)
for method in results:
    if 'IndexFlat_kNN' in results[method]:
        recall = results[method]['IndexFlat_kNN'][5]['recall']
        print(f"{method} IndexFlat Recall@5: {recall:.4f}")
        assert recall < 0.99, f"Recall too high! Still using wrong ground truth?"
```

#### Expected Output:
```
MPAD IndexFlat Recall@5: 0.7234  ← Good! (< 1.0)
PCA IndexFlat Recall@5: 0.6823   ← Good! (< 1.0)
...
```

If all recalls are ≈ 1.0, the fix didn't work.

---

## 📅 Summary

- **Issue**: Ground truth was computed on reduced space (wrong!)
- **Fix**: Ground truth now computed on ORIGINAL space (correct!)
- **Impact**: Recall values will be lower but CORRECT
- **Action Required**: Re-run ALL experiments
- **Files Modified**: 
  - `main_program.py`
  - `main_program_optimized.py`
- **Scripts Affected**: All (scalability, ablation, SIFT/Deep)

**Date**: 2025-10-26  
**Severity**: CRITICAL (all previous results invalid)  
**Status**: ✅ FIXED

