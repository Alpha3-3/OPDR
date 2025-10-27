# Approximate Nearest Neighbor Search Evaluation

This directory contains scripts for evaluating MPAD and various baseline dimensionality reduction methods on approximate nearest neighbor search tasks.

## Core Files

```
Approximate/
├── data_preprocessing.py      # Data loading and preprocessing
├── main_program.py            # Main evaluation program
├── ablation_study.py          # Ablation study scripts
├── scalability_test.py        # Scalability testing
├── plot_scalability_results.py # Results visualization
├── test_complete.py           # Comprehensive test suite
├── README.md                  # This file
└── Result/                    # Output directory
    ├── cache/                 # Cached reduced data and results
    └── scalability_fasttext/  # Scalability test results
```

## Datasets

- **Fasttext**: Word embeddings (300D) - subsampled to 1%, 5%, 10%
- **Isolet**: Speech recognition data (617D) - 80/20 train/test split
- **PBMC3k**: Single-cell RNA-seq data (1838D) - 80/20 train/test split  
- **Arcene**: Microarray data (10000D) - 80/20 train/test split

## Methods

### Dimensionality Reduction (11 methods)
- MPAD, PCA, UMAP, Isomap, Kernel PCA, Random Projection, NMF, LLE, Feature Agglomeration, Autoencoder, VAE

### Index Methods (5 methods)
- IndexFlat_kNN (exact), HNSWFlat, IVFPQ, IVF_PQR, IVF_OPQ_PQ

## Quick Start

### 1. Test System
```bash
python test_complete.py
```

### 2. Run Main Evaluation
```bash
python main_program.py
```

### 3. Run Scalability Test
```bash
python scalability_test.py
```

### 4. Generate Plots
```bash
python plot_scalability_results.py
```

### 5. Run Ablation Study
```bash
python ablation_study.py
```

## Key Features

- **Caching**: Reduced data and k-NN results are automatically cached
- **Memory Monitoring**: Real-time memory usage tracking
- **Parallel Processing**: Multiprocessing support for faster evaluation
- **Comprehensive Testing**: All methods tested with detailed timing and accuracy metrics
- **Visualization**: Automated plotting of scalability results

## Output Structure

```
Result/
├── cache/
│   └── {dataset}_{method}/
│       ├── train_reduced_TD{target_dim}.npy
│       ├── test_reduced_TD{target_dim}.npy
│       ├── ground_truth_k{max_k}.npy
│       └── {index_method}_k{k}_indices.npy
├── scalability_fasttext/
│   ├── results_*.csv
│   ├── summary_*.csv
│   ├── recall_plots.png
│   ├── runtime_plots.png
│   └── memory_plots.png
└── *.csv (evaluation results)
```

## Dependencies

- numpy, pandas, scipy, scikit-learn
- umap-learn, faiss-cpu, tensorflow
- matplotlib, seaborn, tqdm, psutil

## Notes

- All data points are L2-normalized
- Fixed random seeds (seed=1) for reproducibility
- Ground truth computed on original data, compared against reduced data results
- Results include Recall@k, runtime, and memory usage metrics