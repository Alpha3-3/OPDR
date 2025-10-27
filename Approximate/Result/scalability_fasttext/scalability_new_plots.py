import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Use the same style as Ablation Study
ggs = {'font.size': 14}
plt.rcParams.update(ggs)
plt.style.use('tableau-colorblind10')

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
df = pd.read_csv(os.path.join(script_dir, 'scalability_results_optimized.csv'))

# Define methods to plot (5 methods only, excluding LLE)
methods_to_plot = ['MPAD', 'UMAP', 'FeatureAgglomeration', 'RandomProjection', 'NMF']

# Color mapping (from Ablation Study)
method_colors = {
    'MPAD': 'red',
    'UMAP': '#8B4513',
    'FeatureAgglomeration': '#FFD700',
    'RandomProjection': '#808080',
    'LLE': '#4169E1',
    'NMF': '#006400'
}

# Display name mapping (rename MPAD to QPAD in plots)
method_display_names = {
    'MPAD': 'QPAD',  # Changed from MPAD to QPAD
    'UMAP': 'UMAP',
    'FeatureAgglomeration': 'FeatAgg',
    'RandomProjection': 'RandProj',
    'LLE': 'LLE',
    'NMF': 'NMF'
}

# Marker styles (from Ablation Study)
method_markers = {
    'MPAD': {'marker': '*', 'linestyle': '-', 'markersize': 12},  # Star for MPAD
    'UMAP': {'marker': '^', 'linestyle': '-.'},
    'FeatureAgglomeration': {'marker': 's', 'linestyle': '-.'},
    'RandomProjection': {'marker': 'o', 'linestyle': '--'},  # Changed from * to o
    'LLE': {'marker': 'p', 'linestyle': ':'},
    'NMF': {'marker': 'h', 'linestyle': '-'}
}


# Map subsample to set size
subsample_to_size = {
    '01pct': 10000,
    '05pct': 50000,
    '10pct': 100000
}

# Index methods
index_methods = ['IndexFlat_kNN', 'HNSWFlat', 'IVFPQ', 'IVF_PQR', 'IVF_OPQ_PQ']
index_display_names = {
    'IndexFlat_kNN': 'kNN',
    'HNSWFlat': 'HNSW',
    'IVFPQ': 'IVFPQ',
    'IVF_PQR': 'IVF-PQR',
    'IVF_OPQ_PQ': 'IVF-OPQ-PQ'
}

# Prepare data - add set_size column
df['set_size'] = df['subsample'].map(subsample_to_size)

print("=== Creating Recall@10 vs Set Size Plot ===")

# Figure 1: Recall@10 vs Set Size (5 index methods)
fig1, axes1 = plt.subplots(1, 5, figsize=(20, 4.5))
all_handles = {}

for i, idx_method in enumerate(index_methods):
    ax = axes1[i]
    recall_col = f'{idx_method}_recall@10'
    
    for method in methods_to_plot:
        # Filter data for this method
        method_df = df[df['method'] == method].sort_values('set_size')
        
        if len(method_df) == 0 or method_df[recall_col].isna().all():
            continue
        
        # Plot
        color = method_colors.get(method, 'black')
        marker = method_markers.get(method, {}).get('marker', 'o')
        linestyle = method_markers.get(method, {}).get('linestyle', '-')
        label = method_display_names.get(method, method)
        
        line, = ax.plot(
            method_df['set_size'], 
            method_df[recall_col],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
            linewidth=2,
            markersize=8
        )
        all_handles[label] = line
    
    # Set labels and title
    ax.set_xlabel('Set Size', fontsize=16)
    if i == 0:
        ax.set_ylabel('Recall@10', fontsize=16)
    
    ax.set_title(index_display_names[idx_method], fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.set_xticks([10000, 50000, 100000])
    ax.set_xticklabels(['10K', '50K', '100K'], fontsize=12)
    
    # Format y-axis - Set range to 0-0.6
    ax.set_ylim([0, 0.6])
    plt.setp(ax.get_yticklabels(), fontsize=12)

# Add legend
fig1.legend(
    all_handles.values(), 
    all_handles.keys(),
    loc='upper center',
    ncol=len(all_handles),
    fontsize=16,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
# fig1_path = os.path.join(script_dir, 'recall_vs_setsize.png')
# plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
# print(f">>> Saved: {fig1_path}")
plt.show()
print(">>> Figure 1 displayed")

# Figure 2: Runtime Ratio (as scatter plot)
print("\n=== Creating Relative Runtime vs Set Size Plot ===")

# Calculate relative runtime for each method and subsample (10K as baseline=1)
runtime_data = []
for method in methods_to_plot:
    method_df = df[df['method'] == method]
    
    # Get runtime for each subsample
    runtime_01pct = method_df[method_df['subsample'] == '01pct']['dr_time'].values
    runtime_05pct = method_df[method_df['subsample'] == '05pct']['dr_time'].values
    runtime_10pct = method_df[method_df['subsample'] == '10pct']['dr_time'].values
    
    if len(runtime_01pct) > 0 and runtime_01pct[0] > 0:
        base_time = runtime_01pct[0]
        
        # Calculate relative runtime (all methods at 10K = 1.0)
        rel_runtime_10k = 1.0
        rel_runtime_50k = runtime_05pct[0] / base_time if len(runtime_05pct) > 0 and runtime_05pct[0] > 0 else np.nan
        rel_runtime_100k = runtime_10pct[0] / base_time if len(runtime_10pct) > 0 and runtime_10pct[0] > 0 else np.nan
        
        runtime_data.append({
            'method': method,
            'display_name': method_display_names[method],
            'set_size': [10000, 50000, 100000],
            'relative_runtime': [rel_runtime_10k, rel_runtime_50k, rel_runtime_100k]
        })

# Create scatter plot
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
all_handles = {}

for method_data in runtime_data:
    method = method_data['method']
    color = method_colors.get(method, 'black')
    marker = method_markers.get(method, {}).get('marker', 'o')
    markersize = method_markers.get(method, {}).get('markersize', 8)
    label = method_data['display_name']
    
    scatter, = ax2.plot(
        method_data['set_size'],
        method_data['relative_runtime'],
        marker=marker,
        linestyle='-',
        color=color,
        label=label,
        linewidth=2,
        markersize=markersize
    )
    all_handles[label] = scatter

# Add horizontal line at ratio=1
ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

# Labels and formatting
ax2.set_xlabel('Set Size', fontsize=16)
ax2.set_ylabel('Relative Runtime', fontsize=16)
ax2.set_xticks([10000, 50000, 100000])
ax2.set_xticklabels(['10K', '50K', '100K'], fontsize=12)
ax2.legend(fontsize=14, loc='upper left')
ax2.grid(True, alpha=0.3)
plt.setp(ax2.get_yticklabels(), fontsize=12)

plt.tight_layout()
# fig2_path = os.path.join(script_dir, 'runtime_ratio_vs_setsize.png')
# plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
# print(f">>> Saved: {fig2_path}")
plt.show()
print(">>> Figure 2 displayed")

print("\n=== Summary ===")
print("Displayed 2 plots - you can manually adjust and save them")

