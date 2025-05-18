import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Apply global style and font size
plt.style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 18})

# -------------------------------
# 1. Define CSV paths and dataset names.
# -------------------------------
csv_paths = [
    "parameter_sweep_results_Fasttext_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_Isolet_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_Arcene_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_PBMC3k_Multiple_methods_with_additional_baselines.csv"
]
dataset_names = [
    "Fasttext",
    "Isolet",
    "Arcene",
    "PBMC3k"
]

# -------------------------------
# 2. Define base method styles and prepare color cycling.
#    These can be extended with more known methods.
# -------------------------------
method_styles = {
    'MPAD Accuracy':    {'color': 'red', 'short_name': 'MPAD'},
    'UMAP Accuracy':    {'color': 'green', 'short_name': 'UMAP'},
    'Isomap Accuracy':  {'color': 'orange', 'short_name': 'Isomap'},
    'KernelPCA Accuracy': {'color': 'purple', 'short_name': 'KPCA'},
    'MDS Accuracy':     {'color': 'gray', 'short_name': 'MDS'},
    # Example for a potential new method if it becomes common:
    # 'Autoencoder Accuracy': {'color': 'blue', 'short_name': 'AE'}
}

# Get default color cycle from matplotlib's current style
default_color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Filter out colors already used in method_styles to prioritize unique colors for new methods
predefined_colors = {style['color'] for style in method_styles.values() if 'color' in style}
unique_default_colors = [color for color in default_color_palette if color not in predefined_colors]


# -------------------------------
# 3. Create a 2x2 subplot layout for the 4 datasets.
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(15, 10)) # Increased figure size for better layout
axs = axs.flatten()  # flatten to 1D array for easy iteration

# -------------------------------
# 4. Process each dataset and plot the grouped bar chart.
# -------------------------------
for i, csv_path in enumerate(csv_paths):
    ax = axs[i]
    ax.set_title(dataset_names[i]) # Set title early

    try:
        # Load and filter the data (ignore rows with b = 40 or 50).
        df = pd.read_csv(csv_path)
        df = df[~df['b'].isin([40, 50])]
    except FileNotFoundError:
        print(f"Warning: CSV file not found at {csv_path}. Skipping this dataset.")
        ax.text(0.5, 0.5, "CSV file not found", ha='center', va='center')
        continue
    except Exception as e:
        print(f"Warning: Error loading {csv_path}: {e}. Skipping this dataset.")
        ax.text(0.5, 0.5, "Error loading CSV", ha='center', va='center')
        continue


    # 1. Automatically recognize baseline methods from CSV header
    all_csv_cols = df.columns.tolist()
    # Filter for columns that seem to represent accuracies
    # Exclude PCA and FastICA
    discovered_accuracy_cols = [col for col in df.columns if col.endswith("Accuracy") and col != "PCA Accuracy" and col != "FastICA Accuracy"]

    if not discovered_accuracy_cols:
        print(f"Warning: No 'Accuracy' columns found in {csv_path}. Skipping this dataset plot.")
        ax.text(0.5, 0.5, "No accuracy data found", ha='center', va='center')
        continue

    # Prepare plot attributes (name, color, short_name) for discovered methods
    # Reset color index for each plot to ensure consistency if some methods are missing
    color_cycler_index = 0

    current_plot_attributes = []
    for col_name in discovered_accuracy_cols:
        short_name_default = col_name.replace(" Accuracy", "")
        assigned_color = None
        final_short_name = short_name_default

        if col_name in method_styles:
            style_info = method_styles[col_name]
            final_short_name = style_info.get('short_name', short_name_default)
            assigned_color = style_info.get('color')

        if assigned_color is None: # Not in predefined styles or no color specified in style
            if color_cycler_index < len(unique_default_colors):
                assigned_color = unique_default_colors[color_cycler_index]
            else: # Fallback: cycle through all default palette colors if unique ones are exhausted
                assigned_color = default_color_palette[(color_cycler_index - len(unique_default_colors)) % len(default_color_palette)]
            color_cycler_index += 1

        current_plot_attributes.append({
            'full_name': col_name,
            'short_name': final_short_name,
            'color': assigned_color
        })

    # Extract lists for convenience in pandas operations and plotting
    accuracy_cols_to_use = [attr['full_name'] for attr in current_plot_attributes]
    bar_colors_for_plot = [attr['color'] for attr in current_plot_attributes]
    x_tick_labels_for_plot = [attr['short_name'] for attr in current_plot_attributes]

    # Compute group means for each (alpha, b) pair.
    try:
        grouped_param_means = df.groupby(['alpha', 'b'])[accuracy_cols_to_use].mean()
    except KeyError as e:
        print(f"Warning: One of the accuracy columns {e} not found for grouping in {csv_path}. Skipping.")
        ax.text(0.5, 0.5, f"Column {e} not found", ha='center', va='center')
        continue

    if grouped_param_means.empty:
        print(f"Warning: No data after grouping for {csv_path}. Skipping plot.")
        ax.text(0.5, 0.5, "No data after grouping", ha='center', va='center')
        continue

    # Determine the metric/score for selecting the best (alpha, b) combination
    selection_metric_series = None
    if 'MPAD Accuracy' in accuracy_cols_to_use:
        def calculate_mpad_centric_score(row_series):
            mpad_accuracy = row_series['MPAD Accuracy']
            other_method_accuracies = [row_series[col] for col in accuracy_cols_to_use if col != 'MPAD Accuracy']
            if not other_method_accuracies: # MPAD is the only method
                return mpad_accuracy
            return mpad_accuracy - max(other_method_accuracies) # Maximize MPAD's superiority gap
        selection_metric_series = grouped_param_means.apply(calculate_mpad_centric_score, axis=1)
    else:
        # If MPAD is not present, select (alpha, b) that maximizes the highest accuracy of any method
        selection_metric_series = grouped_param_means[accuracy_cols_to_use].max(axis=1)

    if selection_metric_series.empty:
        print(f"Warning: Selection metric series is empty for {csv_path}. Skipping plot.")
        ax.text(0.5, 0.5, "Metric calculation failed", ha='center', va='center')
        continue

    best_params_combination_idx = selection_metric_series.idxmax() # This is an (alpha, b) tuple
    selected_alpha_val, selected_b_val = best_params_combination_idx

    # For the best (alpha, b), get the mean accuracy for each method.
    # This uses the original df to average over any multiple runs for that (alpha,b)
    mask_for_best_params = (df['alpha'] == selected_alpha_val) & (df['b'] == selected_b_val)
    data_at_best_params = df[mask_for_best_params]

    if data_at_best_params.empty:
        print(f"Warning: No data rows for best params alpha={selected_alpha_val}, b={selected_b_val} in {csv_path}. Using grouped mean.")
        # Fallback to already computed mean if original rows are somehow lost/empty
        mean_abs_accuracies = grouped_param_means.loc[best_params_combination_idx]
    else:
        mean_abs_accuracies = data_at_best_params[accuracy_cols_to_use].mean()


    if mean_abs_accuracies.isnull().all():
        print(f"Warning: All mean accuracies are NaN for best params in {csv_path}. Skipping plot.")
        ax.text(0.5, 0.5, "Mean accuracies are NaN", ha='center', va='center')
        continue

    # 2. Compute relative accuracy. Use the method with the highest accuracy as baseline (relative accuracy = 1).
    max_absolute_accuracy = mean_abs_accuracies.max()

    if pd.isna(max_absolute_accuracy) or max_absolute_accuracy == 0:
        # Avoid division by zero or NaN issues if all accuracies are 0 or NaN
        relative_mean_accuracies = mean_abs_accuracies.fillna(0) # Plot NaNs as 0
    else:
        relative_mean_accuracies = mean_abs_accuracies / max_absolute_accuracy

    # Identify the best performing method (which will have relative accuracy of 1)
    # idxmax might fail if all are NaN, but we handled that with max_absolute_accuracy check
    best_performing_method_fullname = mean_abs_accuracies.idxmax() if not mean_abs_accuracies.isnull().all() else None


    # Plot a grouped bar chart of relative accuracies.
    x_indices = np.arange(len(accuracy_cols_to_use))
    bar_width = 0.6  # Consider adjusting if many methods

    bars = ax.bar(x_indices, relative_mean_accuracies, bar_width, color=bar_colors_for_plot)

    # Emphasize the baseline method (highest accuracy) by adding a thick black edge.
    if best_performing_method_fullname and best_performing_method_fullname in accuracy_cols_to_use:
        idx_of_baseline_method = accuracy_cols_to_use.index(best_performing_method_fullname)
        bars[idx_of_baseline_method].set_edgecolor('black')
        bars[idx_of_baseline_method].set_linewidth(2.5)

    # Add a horizontal dotted line at y=1.0 (the baseline relative accuracy).
    ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5)

    # Adjust y-axis limit to ensure the horizontal line and bars are visible.
    current_max_rel_acc = relative_mean_accuracies.max()
    ax.set_ylim(0, max(1.15, current_max_rel_acc * 1.05 if pd.notna(current_max_rel_acc) else 1.15) )


    # Set the x-axis tick labels (use generated short names) horizontally.
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_tick_labels_for_plot, rotation=0, ha='center')
    ax.set_ylabel("Relative Average Accuracy")
    # Title was set at the beginning of the loop

    # Add a text annotation below the plot showing the selected best parameters.
    baseline_text = f"Selected: alpha = {selected_alpha_val}, b = {selected_b_val}"
    ax.text(0.5, -0.22, baseline_text, transform=ax.transAxes, # Adjusted y position for potentially longer x-labels
            ha='center', va='center', fontsize=16) # Slightly smaller font for annotation

# Adjust subplot parameters for overall layout.
plt.subplots_adjust(top=0.94,
                    bottom=0.12, # Increased bottom margin for annotation
                    left=0.07,
                    right=0.98,
                    hspace=0.45, # Increased hspace
                    wspace=0.2)  # Increased wspace

plt.suptitle("Relative Performance of Dimensionality Reduction Methods", fontsize=22, y=0.99)
plt.show()