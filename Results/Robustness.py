import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 18})

# ---------------------------------------------------
# 1. Define CSV paths and dataset names.
# ---------------------------------------------------
csv_paths = [
    "parameter_sweep_results_Fasttext_Multiple_methods.csv",
    "parameter_sweep_results_Isolet_Multiple_methods.csv",
    "parameter_sweep_results_Arcene_Multiple_methods.csv",
    "parameter_sweep_results_PBMC3k_Multiple_methods.csv"
]
dataset_names = [
    "Fasttext",
    "Isolet",
    "Arcene",
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
    'MPAD Accuracy':     {'color': 'red'},      # Emphasize in red
    'UMAP Accuracy':     {'color': 'green'},
    'Isomap Accuracy':   {'color': 'orange'},
    'KernelPCA Accuracy':{'color': 'purple'},
    'MDS Accuracy':      {'color': 'gray'}
}

# ---------------------------------------------------
# 3. Create a 2x2 subplot layout for the 4 datasets.
# ---------------------------------------------------
plt.style.use('tableau-colorblind10')
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
axs = axs.flatten()  # Flatten to 1D array for easy iteration

# ---------------------------------------------------
# 4. Process each dataset and plot the stacked bar chart.
# ---------------------------------------------------
for i, csv_path in enumerate(csv_paths):
    # Load and filter the data (ignore rows with b = 40 or 50).
    df = pd.read_csv(csv_path)
    df = df[~df['b'].isin([40, 50])]

    # Initialize counters for best and second-best occurrences.
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

    # Plot a stacked bar chart.
    ax = axs[i]
    x = np.arange(len(methods))
    width = 0.6

    # Bottom part: "Best" counts.
    bars_best = ax.bar(x, counts_df['Best Count'], width,
                       color=[method_styles[m]['color'] for m in methods])
    # Top part: "Second Best" counts, stacked on top.
    bars_second = ax.bar(x, counts_df['Second Best Count'], width,
                         bottom=counts_df['Best Count'], color='lightgray')

    # Emphasize MPAD Accuracy by adding a thick black border to its best portion.
    mpad_idx = methods.index('MPAD Accuracy')
    bars_best[mpad_idx].set_edgecolor('black')
    bars_best[mpad_idx].set_linewidth(2.5)

    # Set the x-axis tick labels (remove " Accuracy" suffix).
    method_labels = ['MPAD', 'UMAP', 'Isomap', 'KPCA', 'MDS']
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=0, ha='center', fontsize=16)
    ax.set_ylabel("Count", fontsize=18)
    ax.set_title(dataset_names[i], fontsize=18)

# ---------------------------------------------------
# 5. Create one global legend.
# ---------------------------------------------------
legend_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
           markeredgecolor='black', markersize=10,
           label='Best Count (bar color indicates method; MPAD is emphasized)'),
    Line2D([0], [0], marker='s', color='lightgray', markersize=10,
           label='Second Best Count')
]

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.legend(handles=legend_handles, loc='upper center', ncol=1, fontsize=16, bbox_to_anchor=(0.5, 0.99))

# Adjust subplot parameters as specified.
plt.subplots_adjust(top=0.843,
                    bottom=0.049,
                    left=0.086,
                    right=0.995,
                    hspace=0.335,
                    wspace=0.239)

plt.show()
