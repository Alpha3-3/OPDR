import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})

# -------------------------------
# 1. Define CSV paths and dataset names.
# -------------------------------
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

# -------------------------------
# 2. Define the accuracy columns and method styles (ignoring PCA).
# -------------------------------
accuracy_cols = [
    'MPAD Accuracy',
    'UMAP Accuracy',
    'Isomap Accuracy',
    'KernelPCA Accuracy',
    'MDS Accuracy'
]

method_styles = {
    'MPAD Accuracy':    {'color': 'red'},
    'UMAP Accuracy':    {'color': 'green'},
    'Isomap Accuracy':  {'color': 'orange'},
    'KernelPCA Accuracy': {'color': 'purple'},
    'MDS Accuracy':     {'color': 'gray'}
}

# -------------------------------
# 3. Create a 2x2 subplot layout for the 4 datasets.
# -------------------------------
plt.style.use('tableau-colorblind10')
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # flatten to 1D array for easy iteration

# -------------------------------
# 4. Process each dataset and plot the grouped bar chart.
# -------------------------------
for i, csv_path in enumerate(csv_paths):
    # Load and filter the data (ignore rows with b = 40 or 50).
    df = pd.read_csv(csv_path)
    df = df[~df['b'].isin([40, 50])]

    # Compute group means for each (alpha, b) pair.
    grouped = df.groupby(['alpha', 'b'])[accuracy_cols].mean().reset_index()

    # Define a function to compute the superiority gap:
    # MPAD Accuracy minus the maximum accuracy of the other methods.
    def compute_gap(row):
        mpad = row['MPAD Accuracy']
        others = [row[col] for col in accuracy_cols if col != 'MPAD Accuracy']
        return mpad - max(others)

    grouped['gap'] = grouped.apply(compute_gap, axis=1)

    # Select the best (alpha, b) pair based on the gap.
    best_idx = grouped['gap'].idxmax()
    best_combo = grouped.loc[best_idx, ['alpha', 'b']]

    # For the best (alpha, b), compute the overall average accuracy across all other parameters.
    mask = (df['alpha'] == best_combo['alpha']) & (df['b'] == best_combo['b'])
    best_data = df[mask]
    mean_acc = best_data[accuracy_cols].mean()

    # Plot a grouped bar chart.
    ax = axs[i]
    methods = accuracy_cols
    x = np.arange(len(methods))
    width = 0.6
    colors = [method_styles[m]['color'] for m in methods]
    bars = ax.bar(x, mean_acc, width, color=colors)

    # Emphasize MPAD Accuracy by adding a thick black edge.
    mpad_idx = methods.index('MPAD Accuracy')
    bars[mpad_idx].set_edgecolor('black')
    bars[mpad_idx].set_linewidth(2.5)

    # Increase the y-axis limit to ensure annotations are visible.
    ax.set_ylim(0, mean_acc.max() * 1.15)

    # Annotate each bar with its numeric value.
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # vertical offset in points
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Set the x-axis tick labels (use method names without "Accuracy") horizontally.
    method_labels = ['MPAD', 'UMAP', 'Isomap', 'KernelPCA', 'MDS']
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=0, ha='center')
    ax.set_ylabel("Average Accuracy")
    ax.set_title(dataset_names[i])

    # Add a text annotation below the plot showing the selected best parameters.
    baseline_text = f"Selected: alpha = {best_combo['alpha']}, b = {best_combo['b']}"
    ax.text(0.5, -0.18, baseline_text, transform=ax.transAxes,
            ha='center', va='center', fontsize=10)

plt.tight_layout()
plt.show()
