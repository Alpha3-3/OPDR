# Scalability Test Optimization

## Skip Slow Methods for Large Datasets

### Problem
During scalability testing, some dimensionality reduction methods become extremely slow on larger datasets:
- **Isomap**: O(N³) complexity for eigenvalue decomposition
- **KernelPCA**: O(N³) complexity for kernel matrix computation
- **LLE (Locally Linear Embedding)**: O(N²) to O(N³) complexity

For Fasttext 5% (~4K samples) and 10% (~8K samples), these methods can take **hours** to complete a single run.

### Solution
Automatically skip slow methods (`Isomap`, `KernelPCA`, `LLE`) for larger datasets (5% and 10% subsamples).

### Implementation

#### 1. Added `skip_methods` Parameter to `main_evaluation_optimized()`
```python
def main_evaluation_optimized(dataset_name: str,
                              train_file: str,
                              test_file: str,
                              target_dim: int,
                              b_percentage: float,
                              alpha: float,
                              k_values: list,
                              save_results: bool = True,
                              output_dir: str = "Result",
                              skip_methods: list = None):  # NEW PARAMETER
```

#### 2. Modified `scalability_test_optimized.py`
```python
# Determine which methods to skip based on dataset size
skip_methods = []
if subsample in ['05pct', '10pct']:
    skip_methods = ['Isomap', 'KernelPCA', 'LLE']
    print(f"\n[INFO] Large dataset ({subsample}): skipping slow methods: {', '.join(skip_methods)}\n")

# Pass skip_methods to evaluation
all_results, detailed_file, summary_file = main_evaluation_optimized(
    dataset_name=f"Fasttext_{subsample}",
    train_file=train_file,
    test_file=test_file,
    target_dim=target_dim,
    b_percentage=b_percentage,
    alpha=alpha,
    k_values=k_values,
    save_results=True,
    output_dir="Result/scalability_fasttext_optimized",
    skip_methods=skip_methods  # NEW PARAMETER
)
```

### Behavior

#### 1% Subsample (~800 samples)
- **All methods run**: MPAD, PCA, UMAP, Isomap, KernelPCA, RandomProjection, NMF, LLE, FeatureAgglomeration, Autoencoder, VAE
- **Total methods**: 11
- **Estimated time**: ~10-15 minutes

#### 5% Subsample (~4K samples)
- **Skipped**: Isomap, KernelPCA, LLE
- **Remaining**: MPAD, PCA, UMAP, RandomProjection, NMF, FeatureAgglomeration, Autoencoder, VAE
- **Total methods**: 8
- **Estimated time**: ~20-30 minutes (would be 2-3 hours without skipping)

#### 10% Subsample (~8K samples)
- **Skipped**: Isomap, KernelPCA, LLE
- **Remaining**: MPAD, PCA, UMAP, RandomProjection, NMF, FeatureAgglomeration, Autoencoder, VAE
- **Total methods**: 8
- **Estimated time**: ~40-60 minutes (would be 5-8 hours without skipping)

### Output Example

```bash
$ python scalability_test_optimized.py

================================================================================
Testing 01pct subsample
================================================================================
# All 11 methods run normally

================================================================================
Testing 05pct subsample
================================================================================

[INFO] Large dataset (05pct): skipping slow methods: Isomap, KernelPCA, LLE

MAIN EVALUATION - OPTIMIZED MPAD
================================================================================
Dataset: Fasttext_05pct
Target Dimension: 128
MPAD Parameters: b=1.0%, alpha=0.1
k values: [1, 10, 50]
Skipping methods: Isomap, KernelPCA, LLE
================================================================================

[SKIP] Isomap (excluded for this dataset size)
[SKIP] KernelPCA (excluded for this dataset size)
[SKIP] LLE (excluded for this dataset size)

# Remaining 8 methods run normally
```

### CSV Output

The generated CSV will have:
- **1% subsample**: 11 methods with complete data
- **5% subsample**: 8 methods (Isomap, KernelPCA, LLE missing)
- **10% subsample**: 8 methods (Isomap, KernelPCA, LLE missing)

This is intentional and expected. When analyzing results, note that these three methods are not evaluated for larger datasets due to computational constraints.

### Time Savings

| Subsample | Without Skip | With Skip | Time Saved |
|-----------|--------------|-----------|------------|
| 1%        | ~15 min      | ~15 min   | 0 min      |
| 5%        | ~3 hours     | ~30 min   | ~2.5 hours |
| 10%       | ~8 hours     | ~60 min   | ~7 hours   |
| **Total** | **~11 hours**| **~1.75 hours** | **~9.25 hours (84%)** |

### Customization

To modify which methods are skipped or for which dataset sizes:

```python
# In scalability_test_optimized.py, line ~75-78
skip_methods = []
if subsample in ['05pct', '10pct']:  # Change these conditions
    skip_methods = ['Isomap', 'KernelPCA', 'LLE']  # Change these methods
```

Common alternatives:
```python
# Skip only for 10% subsample
if subsample == '10pct':
    skip_methods = ['Isomap', 'KernelPCA', 'LLE', 'UMAP']  # Add UMAP too

# Skip different methods
if subsample in ['05pct', '10pct']:
    skip_methods = ['Isomap', 'LLE']  # Keep KernelPCA

# Never skip (original behavior)
skip_methods = []  # Empty list = no skipping
```

### Testing

To verify the skip functionality works:
```bash
python test_skip_methods.py
```

Expected output:
```
[SKIP] Isomap (excluded for this dataset size)
[SKIP] KernelPCA (excluded for this dataset size)
[SKIP] LLE (excluded for this dataset size)
[OK] Isomap was correctly skipped
[OK] KernelPCA was correctly skipped
[OK] LLE was correctly skipped
```

### Notes

1. **Ablation Study**: This optimization is specific to scalability testing. Ablation studies typically use a single fixed dataset size, so all methods run normally.

2. **Other Datasets**: If you want to use this for other datasets (Isolet, PBMC3k, Arcene), similar logic can be added to their respective test scripts.

3. **Alternative Implementations**: For very large datasets, consider also skipping:
   - `UMAP`: Can be slow on 10K+ samples
   - `Autoencoder`/`VAE`: If training epochs are high

4. **Result Analysis**: When comparing methods across subsamples, note that Isomap, KernelPCA, and LLE only have data for the 1% subsample. This is expected and reflects real-world constraints.

---

**Date**: 2025-10-26  
**Modified Files**:
- `Approximate/main_program_optimized.py`
- `Approximate/scalability_test_optimized.py`
- `Approximate/test_skip_methods.py` (new test)

