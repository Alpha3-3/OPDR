#!/usr/bin/env python3
"""
Plot script for scalability results
Creates three figures: Recall@k, Runtime, and Memory Usage

NOTE on Recall@k calculation:
- Ground truth is obtained by running IndexFlat k-NN on the original (unreduced) data
- This ground truth is then compared against:
  1. IndexFlat k-NN on reduced data
  2. HNSWFlat on reduced data
  3. IVFPQ on reduced data
  4. IVF_PQR on reduced data
  5. IVF_OPQ_PQ on reduced data
- Recall@k = (number of common neighbors / k) for each query point, averaged over all queries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_scalability_results(result_dir="Result/scalability_fasttext"):
    """Load all scalability result files"""
    files = glob.glob(f"{result_dir}/results_*.csv")
    
    all_data = []
    for file in sorted(files):
        df = pd.read_csv(file)
        # Extract subsample size from filename
        if '01pct' in file:
            df['sample_size'] = 9999  # Approximate 1% sample
        elif '05pct' in file:
            df['sample_size'] = 49999  # Approximate 5% sample
        elif '10pct' in file:
            df['sample_size'] = 99999  # Approximate 10% sample
        
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def create_recall_plots(df, output_dir="Result/scalability_fasttext"):
    """Create Recall@k plots"""
    k_values = [1, 10, 50]
    index_methods = ['IndexFlat_kNN', 'HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']
    
    fig, axes = plt.subplots(len(k_values), len(index_methods), figsize=(25, 15))
    fig.suptitle('Recall@k Performance', fontsize=16, fontweight='bold')
    
    methods_to_plot = ['MPAD', 'PCA', 'UMAP', 'Isomap', 'RandomProjection']
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_to_plot)))
    
    for i, k in enumerate(k_values):
        for j, index_method in enumerate(index_methods):
            ax = axes[i, j]
            
            for method, color in zip(methods_to_plot, colors):
                method_data = df[(df['method'] == method) & 
                                  (df['index_method'] == index_method) & 
                                  (df['k'] == k)]
                
                if not method_data.empty:
                    method_data = method_data.sort_values('sample_size')
                    ax.plot(method_data['sample_size'], method_data['recall_at_k'], 
                           marker='o', label=method, color=color, linewidth=2, markersize=8)
            
            if i == 0:
                ax.set_title(index_method, fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'k={k}\nRecall@k', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Sample Size', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recall_plots.png', dpi=300, bbox_inches='tight')
    print(f"[SAVE] Recall plots saved: {output_dir}/recall_plots.png")
    plt.close()

def create_runtime_plots(df, output_dir="Result/scalability_fasttext"):
    """Create Runtime plots"""
    k_values = [1, 10, 50]
    index_methods = ['IndexFlat_kNN', 'HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']
    
    fig, axes = plt.subplots(len(k_values), len(index_methods), figsize=(25, 15))
    fig.suptitle('Runtime Performance', fontsize=16, fontweight='bold')
    
    methods_to_plot = ['MPAD', 'PCA', 'UMAP', 'Isomap', 'RandomProjection']
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_to_plot)))
    
    for i, k in enumerate(k_values):
        for j, index_method in enumerate(index_methods):
            ax = axes[i, j]
            
            for method, color in zip(methods_to_plot, colors):
                method_data = df[(df['method'] == method) & 
                                  (df['index_method'] == index_method) & 
                                  (df['k'] == k)]
                
                if not method_data.empty:
                    method_data = method_data.sort_values('sample_size')
                    ax.plot(method_data['sample_size'], method_data['total_time'], 
                           marker='o', label=method, color=color, linewidth=2, markersize=8)
            
            if i == 0:
                ax.set_title(index_method, fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'k={k}\nRuntime (s)', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Sample Size', fontsize=10)
            ax.set_yscale('log')  # Log scale for time
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/runtime_plots.png', dpi=300, bbox_inches='tight')
    print(f"[SAVE] Runtime plots saved: {output_dir}/runtime_plots.png")
    plt.close()

def create_memory_plots(df, output_dir="Result/scalability_fasttext"):
    """Create Memory Usage plots"""
    k_values = [1, 10, 50]
    index_methods = ['IndexFlat_kNN', 'HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']
    
    fig, axes = plt.subplots(len(k_values), len(index_methods), figsize=(25, 15))
    fig.suptitle('Memory Usage Performance', fontsize=16, fontweight='bold')
    
    methods_to_plot = ['MPAD', 'PCA', 'UMAP', 'Isomap', 'RandomProjection']
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_to_plot)))
    
    for i, k in enumerate(k_values):
        for j, index_method in enumerate(index_methods):
            ax = axes[i, j]
            
            for method, color in zip(methods_to_plot, colors):
                method_data = df[(df['method'] == method) & 
                                  (df['index_method'] == index_method) & 
                                  (df['k'] == k)]
                
                if not method_data.empty:
                    method_data = method_data.sort_values('sample_size')
                    # Use absolute value for memory
                    memory_values = method_data['total_memory_mb'].abs()
                    ax.plot(method_data['sample_size'], memory_values, 
                           marker='o', label=method, color=color, linewidth=2, markersize=8)
            
            if i == 0:
                ax.set_title(index_method, fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'k={k}\nMemory (MB)', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Sample Size', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_plots.png', dpi=300, bbox_inches='tight')
    print(f"[SAVE] Memory plots saved: {output_dir}/memory_plots.png")
    plt.close()

def main():
    """Main plotting function"""
    print("Loading scalability results...")
    df = load_scalability_results()
    
    print(f"Loaded {len(df)} data points")
    print(f"Methods: {df['method'].unique()}")
    print(f"Index methods: {df['index_method'].unique()}")
    
    print("\nCreating plots...")
    
    create_recall_plots(df)
    create_runtime_plots(df)
    create_memory_plots(df)
    
    print("\nAll plots created successfully!")

if __name__ == "__main__":
    main()

