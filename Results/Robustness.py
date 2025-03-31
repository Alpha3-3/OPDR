import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 14})

# ---------------------------------------------------
# 1. Define CSV paths and dataset names.
# ---------------------------------------------------
csv_paths = [
    "parameter_sweep_results_Fasttext_Multiple_methods.csv",
    "parameter_sweep_results_Isolet_Multiple_methods.csv",
    "parameter_sweep_results_MNIST_Multiple_methods.csv",
    "parameter_sweep_results_PBMC3k_Multiple_methods.csv"
]
dataset_names = [
    "Fasttext",
    "Isolet",
    "MNIST",
    "PBMC3k"
]

# ---------------------------------------------------
# 2. Define the accuracy columns and method styles (ignoring PCA).
# ---------------------------------------------------
accuracy_cols = [
    'MPAD Accuracy',
    'UMAP Accuracy',
    'Isomap Accuracy',
    'KernelPCA Accuracy',
    'MDS Accuracy'
]

method_styles = {
    'MPAD Accuracy':    {'color': 'red'},      # Emphasize in red
    'UMAP Accuracy':    {'color': 'green'},
    'Isomap Accuracy':  {'color': 'orange'},
    'KernelPCA Accuracy': {'color': 'purple'},
    'MDS Accuracy':     {'color': 'gray'}
}

# ---------------------------------------------------
# 3. Create a 2x2 subplot layout for the 4 datasets.
# ---------------------------------------------------
plt.style.use('tableau-colorblind10')
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # Flatten to 1D array for easy iteration

# ---------------------------------------------------
# 4. Process each dataset and plot the grouped bar chart.
# ---------------------------------------------------
for i, csv_path in enumerate(csv_paths):
    # Load and filter the data (ignore rows with b = 40 or 50).
    df = pd.read_csv(csv_path)
    df = df[~df['b'].isin([40, 50])]

    # Initialize counters for best and second best occurrences.
    best_counts = {m: 0 for m in accuracy_cols}
    second_counts = {m: 0 for m in accuracy_cols}
    methods = accuracy_cols  # fixed ordering

    # Loop over each row, rank the methods, and count the best and second best.
    for idx, row in df.iterrows():
        accuracies = row[accuracy_cols]
        ranks = accuracies.rank(method='min', ascending=False)
        best_method = ranks.idxmin()
        second_method = None
        for m in methods:
            if ranks[m] == 2:
                second_method = m
                break
        best_counts[best_method] += 1
        if second_method is not None:
            second_counts[second_method] += 1

    # Create a DataFrame for plotting.
    counts_df = pd.DataFrame({
        'Method': methods,
        'Best Count': [best_counts[m] for m in methods],
        'Second Best Count': [second_counts[m] for m in methods]
    })

    # Plot a grouped bar chart.
    ax = axs[i]
    x = np.arange(len(methods))
    width = 0.35

    # Bar for "Best" counts.
    bars1 = ax.bar(x - width/2, counts_df['Best Count'], width,
                   color=[method_styles[m]['color'] for m in methods])
    # Bar for "Second Best" counts.
    bars2 = ax.bar(x + width/2, counts_df['Second Best Count'], width,
                   color='lightgray')

    # Emphasize MPAD Accuracy by adding a thick black border.
    mpad_idx = methods.index('MPAD Accuracy')
    bars1[mpad_idx].set_edgecolor('black')
    bars1[mpad_idx].set_linewidth(2.5)

    # Set the x-axis tick labels (remove " Accuracy" suffix).
    method_labels = ['MPAD', 'UMAP', 'Isomap', 'KernelPCA', 'MDS']
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=0, ha='center', fontsize=10)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(dataset_names[i], fontsize=14)

# ---------------------------------------------------
# 5. Create one global legend.
# ---------------------------------------------------
# Create custom legend entries.
legend_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
           markeredgecolor='black', markersize=10, label='Best Count (bar color indicates method; MPAD is emphasized)'),
    Line2D([0], [0], marker='s', color='lightgray', markersize=10, label='Second Best Count')
]

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.legend(handles=legend_handles, loc='upper center', ncol=1, fontsize=10, bbox_to_anchor=(0.5, 0.98))
plt.show()
