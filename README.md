# OPDR: Optimized Projection for Dimensionality Reduction

A comprehensive framework for evaluating dimensionality reduction methods on approximate nearest neighbor search tasks, with a focus on MPAD (Metric Preserving Approximate Dimensionality reduction) and various baseline methods.

## 📁 Project Structure

```
.
├── Approximate/              # Main evaluation framework (NEW)
│   ├── main_program.py       # Primary evaluation pipeline
│   ├── ablation_study.py     # Ablation studies
│   ├── scalability_test.py   # Scalability benchmarks
│   ├── mpad_optimized.py     # Optimized MPAD implementation
│   └── Result/               # Experimental results and plots
│
├── Batch Process/            # Parallel batch processing scripts
│   ├── Sequential.py         # Sequential evaluation
│   └── Parallel_*.py         # Dataset-specific parallel runs
│
├── Results/                  # Analysis and visualization scripts
│   ├── Best Representative.py
│   ├── Ablation Study.py
│   ├── Overview.py
│   └── Robustness.py
│
├── Scalability/              # Scalability analysis tools
│   ├── Scalability.py
│   └── Indexability.py
│
├── More Baseline/            # Additional baseline comparisons
│   └── More Baseline.py
│
├── Dataset processing and testing/  # Dataset preparation utilities
│   ├── Arcene/
│   ├── Fasttext/
│   ├── Isolet/
│   ├── PBMC3k/
│   └── ...
│
└── Archive/                  # Archived experiments and legacy code
    ├── Approximate_Archive/  # Old test scripts and documentation
    └── Methods/              # Legacy method implementations
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn
pip install umap-learn faiss-cpu tensorflow
pip install matplotlib seaborn tqdm psutil
```

### Run Main Experiments

**Option 1: Use the new optimized framework (Recommended)**
```bash
cd Approximate
python main_program.py
```

**Option 2: Use batch processing**
```bash
cd "Batch Process"
python Sequential.py
```

### Run Specific Analyses

```bash
# Ablation study
cd Approximate
python ablation_study.py

# Scalability tests
python scalability_test.py

# Generate visualizations
cd ../Results
python "Best Representative.py"
```

## 📊 Datasets

The framework supports multiple high-dimensional datasets:

| Dataset | Dimensions | Train/Test | Description |
|---------|-----------|------------|-------------|
| **Fasttext** | 300 | Various sizes | Word embeddings (1%, 5%, 10%, 100%) |
| **Isolet** | 617 | 6238/1559 | Speech recognition features |
| **PBMC3k** | 1838 | 2400/600 | Single-cell RNA-seq data |
| **Arcene** | 10000 | 80/20 split | Microarray cancer data |
| **SIFT1M** | 128 | 1M/10K | Image descriptors |
| **Deep10M** | 96 | 10M queries | Deep learning features |

## 🔧 Methods Evaluated

### Dimensionality Reduction Methods
- **MPAD** (Our method)
- PCA, Kernel PCA
- UMAP, Isomap, LLE
- Random Projection
- NMF, Feature Agglomeration
- Autoencoder, VAE

### Index Methods (via FAISS)
- IndexFlat (exact k-NN)
- HNSW (Hierarchical Navigable Small World)
- IVFPQ (Inverted File with Product Quantization)
- IVF_PQR (with re-ranking)
- IVF_OPQ_PQ (with optimized product quantization)

## 📈 Evaluation Metrics

- **Recall@k**: Accuracy of approximate k-NN search
- **Runtime**: Reduction time + indexing time + query time
- **Memory Usage**: Peak memory consumption
- **Speedup Ratio**: Relative to baseline methods

## 🗂️ Key Features

- ✅ **Comprehensive Evaluation**: 11 DR methods × 5 index methods
- ✅ **Caching System**: Automatic caching of reduced data and results
- ✅ **Parallel Processing**: Multi-core support for faster experiments
- ✅ **Memory Monitoring**: Real-time memory tracking with psutil
- ✅ **Reproducibility**: Fixed random seeds (seed=1)
- ✅ **Visualization**: Automated plot generation

## 📝 Output Files

Results are saved in CSV format with the following structure:

```
results_*.csv         # Detailed per-query results
summary_*.csv         # Aggregated statistics
scalability_*.csv     # Scalability benchmark results
```

## 🔬 Experiment Workflows

### 1. Dataset Preparation
```bash
cd "Dataset processing and testing/{dataset}"
python "Gen Random Pts.py"
python Inspect.py  # Verify generated .npy files
```

### 2. Main Evaluation
```bash
cd Approximate
python main_program.py
```

### 3. Analysis & Visualization
```bash
cd Results
python Overview.py
python "Best Representative.py"
python Robustness.py
```

## 📚 Documentation

- **Approximate/README.md**: Detailed documentation for the evaluation framework
- **Archive/**: Historical experiments and development notes

## 🤝 Contributing

When adding new datasets:
1. Create a directory under `Dataset processing and testing/`
2. Add `Gen Random Pts.py` for data generation
3. Use `Inspect.py` to verify the generated `.npy` files
4. Update dataset configurations in the main scripts

## ⚠️ Notes

- All vectors are L2-normalized before evaluation
- Ground truth is computed on original (unreduced) data
- `.npy` files and `.csv` results are excluded from version control (see `.gitignore`)
- Archived experiments are kept in `Archive/` for reference

## 📖 Citation

If you use this code in your research, please cite our work:

```bibtex
@article{your_paper,
  title={OPDR: Optimized Projection for Dimensionality Reduction},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 📄 License

[Add your license information here]

## 🔗 Related Resources

- GitHub Repository: https://github.com/Alpha3-3/OPDR
- [Add other relevant links]
