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

# Load detailed results data (b=1.0, alpha=0.1)
detailed_df = pd.read_csv(os.path.join(script_dir, 'results_SIFT1M_TD64_b1.0_alpha0.1_20251026_224118.csv'))

# Define methods to plot
methods_to_plot = ['MPAD', 'PCA', 'UMAP', 'FeatureAgglomeration', 'RandomProjection', 'NMF', 'Autoencoder']

# Color mapping (from Ablation Study)
method_colors = {
    'MPAD': 'red',
    'PCA': '#FF8C00',
    'Autoencoder': '#32CD32',
    'RandomProjection': '#808080',
    'FeatureAgglomeration': '#FFD700',
    'UMAP': '#8B4513',
    'NMF': '#006400'
}

# Marker styles (from Ablation Study)
method_markers = {
    'MPAD': {'marker': 'o', 'linestyle': '-'},
    'PCA': {'marker': 'v', 'linestyle': '-'},
    'Autoencoder': {'marker': 'x', 'linestyle': '--'},
    'RandomProjection': {'marker': '*', 'linestyle': '--'},
    'FeatureAgglomeration': {'marker': 's', 'linestyle': '-.'},
    'UMAP': {'marker': '^', 'linestyle': '-.'},
    'NMF': {'marker': 'h', 'linestyle': '-'}
}

# Display names (rename MPAD to QPAD)
method_display_names = {
    'MPAD': 'QPAD',  # Changed from MPAD to QPAD
    'PCA': 'PCA',
    'Autoencoder': 'AE',
    'RandomProjection': 'RandProj',
    'FeatureAgglomeration': 'FeatAgg',
    'UMAP': 'UMAP',
    'NMF': 'NMF'
}

# Index methods (excluding IVFPQ and IVF_PQR)
index_methods = ['IndexFlat_kNN', 'HNSWFlat', 'IVF_OPQ_PQ']
index_display_names = {
    'IndexFlat_kNN': 'kNN',
    'HNSWFlat': 'HNSW',
    'IVF_OPQ_PQ': 'IVF-OPQ-PQ'
}

print("=== Detailed Results (b=1.0, α=0.1) ===")
print("=== Creating Recall@k vs k Plot ===")

# Figure 1: Recall@k vs k (only 3 index methods now)
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))
all_handles = {}

# Get unique k values
k_values = sorted(detailed_df['k'].unique())
print(f"K values: {k_values}")

for i, idx_method in enumerate(index_methods):
    ax = axes1[i]
    
    for method in methods_to_plot:
        # Filter data for this method and index
        method_data = detailed_df[(detailed_df['method'] == method) & 
                                   (detailed_df['index_method'] == idx_method)].sort_values('k')
        
        if len(method_data) == 0:
            continue
        
        # Extract recall values
        recalls = method_data['recall_at_k'].values
        k_vals = method_data['k'].values
        
        # Plot
        color = method_colors.get(method, 'black')
        marker = method_markers.get(method, {}).get('marker', 'o')
        linestyle = method_markers.get(method, {}).get('linestyle', '-')
        label = method_display_names.get(method, method)
        
        line, = ax.plot(
            k_vals, 
            recalls,
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
            linewidth=2,
            markersize=8
        )
        all_handles[label] = line
    
    # Set labels and title
    ax.set_xlabel('k', fontsize=16)
    if i == 0:
        ax.set_ylabel('Recall@k', fontsize=16)
    
    ax.set_title(index_display_names[idx_method], fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(int(k)) for k in k_values], fontsize=12)
    
    # Format y-axis
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
plt.show()
print(">>> Figure 1 displayed (Detailed Results)")

# Figure 2: Runtime vs Method
print("\n=== Creating Runtime vs Method Plot ===")

# Prepare runtime data - get unique runtime for each method
runtime_data = []
for method in methods_to_plot:
    method_rows = detailed_df[detailed_df['method'] == method]
    
    if len(method_rows) > 0:
        # Get the unique dr_time for this method (should be same for all rows of same method)
        runtime = method_rows['dr_time'].iloc[0]
        runtime_data.append({
            'method': method,
            'display_name': method_display_names[method],
            'runtime': runtime
        })

runtime_df = pd.DataFrame(runtime_data)
runtime_df = runtime_df.sort_values('runtime')  # Sort by runtime

# Create bar plot
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

x = np.arange(len(runtime_df))
width = 0.6

# Plot bars using method colors - FIX: use enumerate on range, not iterrows
bars = []
for idx, row in enumerate(runtime_df.itertuples()):
    method = row.method
    color = method_colors.get(method, 'black')
    
    bar = ax2.bar(idx, row.runtime, width, 
                   color=color, alpha=0.85)
    bars.append(bar)

# Labels and formatting
ax2.set_xlabel('Method', fontsize=16)
ax2.set_ylabel('Runtime (seconds)', fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels(runtime_df['display_name'].values, fontsize=14, rotation=0)
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.get_yticklabels(), fontsize=12)

plt.tight_layout()
plt.show()
print(">>> Figure 2 displayed (Detailed Results)")

print("\n=== Summary (Detailed Results: b=1.0, α=0.1) ===")
print("Displayed 2 plots - you can manually adjust and save them")
print("\nRuntime (seconds):")
print(runtime_df[['display_name', 'runtime']].to_string(index=False))

