import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Load the data
df = pd.read_csv("parameter_sweep_results_Isolet_Multiple_methods100.csv")

# Define the accuracy columns
accuracy_columns = [
    "DW-PMAD Accuracy",
    "PCA Accuracy",
    "UMAP Accuracy",
    "Isomap Accuracy",
    "KernelPCA Accuracy",
    "MDS Accuracy"
]

# Determine the best method and best accuracy for each row.
df["Best Method"] = df[accuracy_columns].idxmax(axis=1)
df["Best Accuracy"] = df[accuracy_columns].max(axis=1)

# Get all unique (alpha, b) parameter combinations.
unique_params = df[['alpha', 'b']].drop_duplicates()
n_combinations = unique_params.shape[0]

# Define a grid layout (approximately square)
n_cols = math.ceil(math.sqrt(n_combinations))
n_rows = math.ceil(n_combinations / n_cols)

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
if n_combinations == 1:
    axs = np.array([axs])  # ensure axs is iterable
axs = axs.flatten()  # flatten in case it's a 2D array

# Loop over each parameter combination and plot in the corresponding subplot.
for idx, (_, params) in enumerate(unique_params.iterrows()):
    alpha_val = params['alpha']
    b_val = params['b']

    # Filter the DataFrame for the current (alpha, b) combination.
    group_df = df[(df['alpha'] == alpha_val) & (df['b'] == b_val)]
    group_df_sorted = group_df.sort_values(by=["Target Ratio", "k"])

    num_rows_group = group_df_sorted.shape[0]
    num_methods = len(accuracy_columns)
    x = np.arange(num_rows_group)  # one group per row in this parameter combo
    width = 0.13  # width for each bar

    ax = axs[idx]

    # Plot a bar for each method.
    for i, col in enumerate(accuracy_columns):
        accuracies = group_df_sorted[col]
        offset = x - width * num_methods / 2 + i * width + width / 2
        bars = ax.bar(offset, accuracies, width, label=col)

        # Highlight the best method for each row.
        for j, bar in enumerate(bars):
            if group_df_sorted.iloc[j]["Best Method"] == col:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
                # If the best method is DW-PMAD, add a red star.
                if col == "DW-PMAD Accuracy":
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,  # adjust vertical offset as needed
                        '*',
                        ha='center', va='bottom',
                        color='red', fontsize=12
                    )

    # Create x-axis tick labels.
    xtick_labels = [f"TR={row['Target Ratio']}\nk={row['k']}" for _, row in group_df_sorted.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel('Accuracy')
    ax.set_title(f"alpha={alpha_val}, b={b_val}")
    ax.legend(fontsize='small')

# Hide any unused subplots.
for ax in axs[n_combinations:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
