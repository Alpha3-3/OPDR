# Parameter Summary for All Experiments

## Date
October 27, 2025

## Overview

This document summarizes the MPAD parameters used across different experiments after optimization.

---

## 1. Scalability Test (Fasttext)

**Script**: `scalability_test_optimized.py`

**Dataset**: Fasttext (1%, 5%, 10% subsamples)

**Parameters**:
- `target_dim` = 128
- `b_percentage` = 4.0% *(optimized from 1.0%)*
- `alpha` = 0.4 *(optimized from 0.1)*
- `k_values` = [1, 10, 50]

**Skipped methods**: Isomap, KernelPCA, LLE (for 5% and 10% only)

**Rationale**: Higher b and alpha for better recall at cost of ~4× slower runtime

**Expected runtime**: 4-8 hours

---

## 2. Large Datasets

**Script**: `run_sift_deep_optimized.py`

### SIFT1M
- `target_dim` = 64
- `b_percentage` = 0.5% *(reduced from 1.0%)*
- `alpha` = 0.4 *(increased from 0.1)*
- `k_values` = [1, 10, 50]
- **Rationale**: Lower b for speed on 1M samples, higher alpha for quality

### Fasttext 100%
- `target_dim` = 128
- `b_percentage` = 4.0%
- `alpha` = 0.4
- `k_values` = [1, 10, 50]
- **Rationale**: Same as scalability test for consistency

### Deep10M
- `target_dim` = 64
- `b_percentage` = 1.0%
- `alpha` = 0.1
- `k_values` = [1, 10, 50]
- **Rationale**: Conservative parameters for 10M samples (too slow otherwise)

**Skipped methods**: Isomap, KernelPCA, LLE, tSNE, VAE

**Expected runtime**: 6-10 hours for all three datasets

---

## 3. Ablation Study

**Script**: `ablation_study_optimized.py`

### Fasttext (1% subsample)
**Base parameters**:
- `target_dim` = 128
- `b_percentage` = 1.0%
- `alpha` = 0.1
- `k_values` = [1, 10, 50]

**Variations**:
- `target_dim` ∈ {64, 128, 192}
- `b_percentage` ∈ {0.5%, 1.0%, 2.0%, 4.0%, 8.0%}
- `alpha` ∈ {0.05, 0.10, 0.20, 0.40}

### Isolet
**Base parameters**:
- `target_dim` = 256
- `b_percentage` = 1.0%
- `alpha` = 0.1
- `k_values` = [1, 10, 50]

**Variations**:
- `target_dim` ∈ {64, 128, 256, 384}
- `b_percentage` ∈ {0.5%, 1.0%, 2.0%, 4.0%, 8.0%}
- `alpha` ∈ {0.05, 0.10, 0.20, 0.40}

### PBMC3k
**Base parameters**:
- `target_dim` = 384
- `b_percentage` = 2.0%
- `alpha` = 0.4
- `k_values` = [1, 10, 50]

**Variations**:
- `target_dim` ∈ {128, 256, 384, 512}
- `b_percentage` ∈ {0.5%, 1.0%, 2.0%, 4.0%, 8.0%}
- `alpha` ∈ {0.05, 0.10, 0.20, 0.40}

### Arcene
**Base parameters**:
- `target_dim` = 512
- `b_percentage` = 4.0%
- `alpha` = 0.4
- `k_values` = [1, 10, 50]

**Variations**:
- `target_dim` ∈ {128, 256, 384, 512, 1024}
- `b_percentage` ∈ {0.5%, 1.0%, 2.0%, 4.0%, 8.0%}
- `alpha` ∈ {0.05, 0.10, 0.20, 0.40}

**Skipped methods**: Isomap, KernelPCA, LLE, tSNE, VAE

**Expected runtime per dataset**: 2-4 hours

**Total runtime for all 4 datasets**: 10-15 hours

---

## Parameter Selection Rationale

### `b_percentage` (Sampling Ratio)

Controls how many point pairs are sampled for objective computation.

| Value | Use Case | Trade-off |
|-------|----------|-----------|
| 0.5% | Very large datasets (SIFT1M) | Fast but may miss structure |
| 1.0% | Default, works for most cases | Balanced speed/quality |
| 2.0% | Medium datasets (PBMC3k) | Better quality, 2× slower |
| 4.0% | Small-medium datasets (Fasttext, Arcene) | High quality, 4× slower |
| 8.0% | Ablation only, not for production | Best quality, 8× slower |

**Rule of thumb**: Higher b for smaller datasets or when quality is critical.

### `alpha` (Orthogonality Penalty)

Controls how strongly we enforce orthogonality between projection axes.

| Value | Use Case | Effect |
|-------|----------|--------|
| 0.05 | Exploratory, weak constraint | More flexible axes |
| 0.10 | Default, light constraint | Balanced |
| 0.20 | Medium constraint | Better axis independence |
| 0.40 | Strong constraint | Highly independent axes |

**Rule of thumb**: Higher alpha for higher target dimensions or when dimensional efficiency matters.

### `target_dim` (Target Dimension)

Based on original dimension and desired compression ratio.

| Dataset | Original Dim | Target Dim | Compression |
|---------|--------------|------------|-------------|
| Fasttext | 300 | 128 | 2.3× |
| Isolet | 617 | 256 | 2.4× |
| PBMC3k | 1838 | 384 | 4.8× |
| Arcene | 10000 | 512 | 19.5× |
| SIFT1M | 128 | 64 | 2× |
| Deep10M | 96 | 64 | 1.5× |

**Rule of thumb**: 
- For n < 1000: TD = n/2 to n/4
- For n > 1000: TD = sqrt(n) to n/10

---

## Methods Skipped Across All Experiments

For efficiency and stability, the following methods are skipped:

1. **Isomap**: O(N³) complexity, too slow for N > 5000
2. **KernelPCA**: O(N²) memory, fails on large datasets
3. **LLE**: Numerically unstable, often fails to converge
4. **tSNE**: Not designed for general DR, only visualization
5. **VAE**: TensorFlow/Keras compatibility issues

**Remaining methods** (8 total):
- MPAD_Optimized ✅
- PCA ✅
- UMAP ✅
- RandomProjection ✅
- FastICA ✅
- NMF ✅
- FeatureAgglomeration ✅
- Autoencoder ✅

---

## Execution Order

For optimal resource usage:

1. **Scalability** (Fasttext 1%, 5%, 10%) - 4-8 hours
2. **Large Datasets** (SIFT1M → Fasttext 100% → Deep10M) - 6-10 hours
3. **Ablation Study** (Fasttext → Isolet → PBMC3k → Arcene) - 10-15 hours

**Total estimated time**: 20-33 hours

Can be run in parallel on separate servers/sessions to reduce wall-clock time.

---

## Quick Reference Commands

### Upload updated files
```bash
cd Approximate
scp scalability_test_optimized.py run_sift_deep_optimized.py ablation_study_optimized.py \
    jiuzhou@er074.utah.cloudlab.us:~/Approximate/
```

### Run experiments
```bash
# Scalability
tmux new -s scalability
python scalability_test_optimized.py

# Large datasets
tmux new -s large_datasets
python run_sift_deep_optimized.py

# Ablation (specify dataset)
tmux new -s ablation_fasttext
python ablation_study_optimized.py Fasttext
```

### Monitor progress
```bash
tmux ls                    # List sessions
tmux attach -t <session>   # Attach to session
# Ctrl+B, D to detach
```

---

**Last Updated**: October 27, 2025  
**Status**: All parameters optimized and tested locally

