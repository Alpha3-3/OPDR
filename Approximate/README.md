# Approximate Nearest Neighbor Search Evaluation Framework

This directory contains the main evaluation pipeline for testing dimensionality reduction methods on approximate k-NN search tasks.

## 📁 Core Files

| File | Description |
|------|-------------|
| `main_program.py` | Main evaluation pipeline |
| `main_program_optimized.py` | Optimized version with improved performance |
| `ablation_study.py` | Ablation study experiments |
| `ablation_study_optimized.py` | Optimized ablation study |
| `data_preprocessing.py` | Data loading and preprocessing utilities |
| `mpad_optimized.py` | Optimized MPAD implementation |
| `mpad_numba_kernels.py` | Numba-optimized kernels for MPAD |
| `scalability_test.py` | Scalability benchmarking |
| `scalability_test_optimized.py` | Optimized scalability tests |
| `plot_scalability_results.py` | Results visualization |

## 🗂️ Directory Structure

```
Approximate/
├── main_program.py
├── main_program_optimized.py
├── ablation_study.py
├── ablation_study_optimized.py
├── scalability_test.py
├── scalability_test_optimized.py
├── mpad_optimized.py
├── mpad_numba_kernels.py
├── data_preprocessing.py
├── plot_scalability_results.py
├── README.md
└── Result/
    ├── Archive/              # Historical results
    ├── scalability_fasttext/ # Fasttext scalability results
    └── scalability_SIFT1M/   # SIFT1M scalability results
```

## 🚀 Quick Start

### 1. Run Complete Evaluation

```bash
# Use optimized version (recommended)
python main_program_optimized.py

# Or use standard version
python main_program.py
```

### 2. Run Ablation Study

```bash
# Test MPAD with different parameter settings
python ablation_study_optimized.py
```

### 3. Run Scalability Tests

```bash
# Test scalability with increasing dataset sizes
python scalability_test_optimized.py
```

### 4. Generate Plots

```bash
# Create visualizations from results
python plot_scalability_results.py
```

## 📊 Supported Datasets

The framework automatically loads datasets from `.npy` files:

- **Fasttext**: `training_vectors_*pct_Fasttext.npy`, `testing_vectors_*pct_Fasttext.npy`
- **Isolet**: `training_vectors_Isolet.npy`, `testing_vectors_Isolet.npy`
- **PBMC3k**: `training_vectors_PBMC3k.npy`, `testing_vectors_PBMC3k.npy`
- **Arcene**: `training_vectors_Arcene.npy`, `testing_vectors_Arcene.npy`
- **SIFT1M**: `training_vectors_SIFT1M.npy`, `testing_vectors_SIFT1M.npy`
- **Deep10M**: `training_vectors_Deep10M.npy`, `testing_vectors_Deep10M.npy`

## 🔧 Methods Evaluated

### Dimensionality Reduction Methods (11 total)
1. **MPAD** (Metric Preserving Approximate Dimensionality reduction)
2. **PCA** (Principal Component Analysis)
3. **UMAP** (Uniform Manifold Approximation and Projection)
4. **Isomap** (Isometric Mapping)
5. **Kernel PCA**
6. **Random Projection**
7. **NMF** (Non-negative Matrix Factorization)
8. **LLE** (Locally Linear Embedding)
9. **Feature Agglomeration**
10. **Autoencoder**
11. **VAE** (Variational Autoencoder)

### Index Methods (5 total)
1. **IndexFlat** (Exact k-NN, baseline)
2. **HNSWFlat** (Hierarchical Navigable Small World)
3. **IVFPQ** (Inverted File with Product Quantization)
4. **IVF_PQR** (IVF + PQ with re-ranking)
5. **IVF_OPQ_PQ** (IVF with Optimized Product Quantization)

## 📈 Key Features

### Caching System
- Reduced data is automatically cached in `Result/cache/{dataset}_{method}/`
- Ground truth k-NN results are cached to avoid recomputation
- Index results are cached per configuration

### Memory Monitoring
- Real-time memory usage tracking with `psutil`
- Peak memory consumption recorded for each method
- Memory plots generated automatically

### Parallel Processing
- Numba JIT compilation for MPAD kernels
- Multi-threaded index building (via FAISS)
- Parallel query processing support

### Reproducibility
- Fixed random seeds (`seed=1`)
- Deterministic operations
- Version tracking in output files

## 📋 Output Structure

```
Result/
├── cache/
│   └── {dataset}_{method}/
│       ├── train_reduced_TD{dim}.npy
│       ├── test_reduced_TD{dim}.npy
│       ├── ground_truth_k{max_k}.npy
│       └── {index}_k{k}_indices.npy
│
├── scalability_fasttext/
│   ├── results_*.csv           # Detailed results
│   ├── recall_vs_setsize.png   # Recall plots
│   └── runtime_ratio_vs_setsize.png
│
└── scalability_SIFT1M/
    ├── consolidated_results.csv
    ├── plot_consolidated_results.py
    └── plot_detailed_results.py
```

## 🔬 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Recall@k** | Fraction of true k-nearest neighbors found |
| **Reduction Time** | Time to reduce dimensions |
| **Index Build Time** | Time to build search index |
| **Query Time** | Time for k-NN queries |
| **Total Runtime** | Sum of all time components |
| **Peak Memory** | Maximum memory usage (MB) |
| **Speedup Ratio** | Runtime relative to baseline |

## ⚙️ Configuration

### MPAD Parameters
- `target_dim`: Target dimensionality (default: 64, 128)
- `b`: Threshold multiplier (default: 1.0, 4.0)
- `alpha`: Learning rate for gradient descent (default: 0.1, 0.4)

### Index Parameters
- `k`: Number of nearest neighbors (default: 10)
- `nlist`: Number of inverted lists for IVF (default: 100)
- `m`: Number of subquantizers for PQ (default: 8)
- `nprobe`: Number of lists to visit during search (default: 10)

### Scalability Test Settings
- Dataset sizes: [500, 1000, 1500, 2000, 2500, 3000, 5000, 8000]
- Target dimensions: [32, 64, 128]
- Multiple parameter combinations for ablation studies

## 📝 Dependencies

```
numpy>=1.19.0
pandas>=1.2.0
scipy>=1.6.0
scikit-learn>=0.24.0
umap-learn>=0.5.0
faiss-cpu>=1.7.0
tensorflow>=2.4.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.50.0
psutil>=5.8.0
numba>=0.53.0
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'faiss'`
- **Solution**: Install FAISS: `pip install faiss-cpu` (or `faiss-gpu` for GPU support)

**Issue**: Out of memory errors
- **Solution**: Reduce `target_dim` or use smaller dataset subsamples

**Issue**: Slow performance
- **Solution**: Use `*_optimized.py` versions with Numba acceleration

**Issue**: UMAP warnings about `n_neighbors`
- **Solution**: These are handled automatically by the framework

## 📖 Usage Examples

### Example 1: Quick Test
```python
# Test MPAD with default parameters
python main_program_optimized.py
```

### Example 2: Custom Ablation Study
```python
# Edit ablation_study_optimized.py to modify:
# - TARGET_DIMS = [32, 64, 128, 256]
# - B_VALUES = [0.5, 1.0, 2.0, 4.0]
# - ALPHA_VALUES = [0.05, 0.1, 0.2, 0.4]

python ablation_study_optimized.py
```

### Example 3: Scalability Analysis
```python
# Test with custom dataset sizes
python scalability_test_optimized.py
```

## 📚 Notes

- All data points are **L2-normalized** before processing
- Ground truth is computed on **original (unreduced) data**
- Recall is measured against ground truth computed on full-dimensional data
- Cached results are automatically reused if parameters match
- Progress bars show real-time status using `tqdm`

## 🔄 Version History

- **v2.0**: Optimized implementations with Numba
- **v1.0**: Initial evaluation framework

## 📞 Support

For issues or questions:
1. Check the main project README: `../README.md`
2. Review archived documentation: `../Archive/Approximate_Archive/`
3. Open an issue on GitHub: https://github.com/Alpha3-3/OPDR

---

**Last Updated**: October 2025
