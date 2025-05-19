import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Apply global style and font size
plt.style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 18})

# -------------------------------
# 1. Define CSV paths, dataset names, and X-axis abbreviations.
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

# Define abbreviations for specific methods to be used on the X-axis
# IMPORTANT: The keys here (e.g., "Feature Agglomeration") must EXACTLY MATCH
# the result of `column_name.strip().replace(" Accuracy", "").strip()`
# from your CSV files for these abbreviations to apply.
xaxis_abbreviations = {
    "Feature Agglomeration": "FAgg",
    "Random Projection": "RP",
    "Autoencoder": "AE"
}

# Methods to specifically debug for x-axis label generation
debug_methods_for_xaxis = set(xaxis_abbreviations.keys())

# -------------------------------
# 2. Define base method styles and prepare color cycling.
# -------------------------------
method_styles = {
    'MPAD Accuracy':    {'color': 'red', 'short_name': 'MPAD'},
    'UMAP Accuracy':    {'color': 'green', 'short_name': 'UMAP'},
    'Isomap Accuracy':  {'color': 'orange', 'short_name': 'Isomap'},
    'KernelPCA Accuracy': {'color': 'purple', 'short_name': 'KPCA'},
    'MDS Accuracy':     {'color': 'gray', 'short_name': 'MDS'},
    # Add styles for your specific methods if they are not covered by the general cycling,
    # e.g., if "Feature Agglomeration Accuracy" should always be a specific color.
    # 'Feature Agglomeration Accuracy': {'color': 'cyan'}, # 'short_name' here will be a fallback if not in xaxis_abbreviations
    # 'Random Projection Accuracy': {'color': 'magenta'},
    # 'Autoencoder Accuracy': {'color': 'brown'},
}

default_color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
predefined_colors_in_styles = {style['color'] for style in method_styles.values() if 'color' in style}
unique_default_colors = [color for color in default_color_palette if color not in predefined_colors_in_styles]

globally_assigned_new_method_colors = {} # Stores color for new base_method_names
global_color_cycler_index = 0

final_legend_info = {} # maps legend label (base_method_name) to its color

# -------------------------------
# 3. Create a 2x2 subplot layout for the 4 datasets.
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(17, 13))
axs = axs.flatten()

# -------------------------------
# 4. Process each dataset and plot the grouped bar chart.
# -------------------------------
print("--- Starting Dataset Processing ---")
for i, csv_path in enumerate(csv_paths):
    ax = axs[i]
    ax.set_title(dataset_names[i], fontsize=20)
    print(f"\nProcessing Dataset: {dataset_names[i]} ({csv_path})")

    try:
        df = pd.read_csv(csv_path)
        df = df[~df['b'].isin([40, 50])]
    except FileNotFoundError:
        print(f"  ERROR: CSV file not found at {csv_path}. Skipping this dataset.")
        ax.text(0.5, 0.5, "CSV file not found", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Data not found)", fontsize=16)
        continue
    except Exception as e:
        print(f"  ERROR: Error loading {csv_path}: {e}. Skipping this dataset.")
        ax.text(0.5, 0.5, "Error loading CSV", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Error loading data)", fontsize=16)
        continue

    discovered_accuracy_cols = [col for col in df.columns if col.endswith("Accuracy") and col != "PCA Accuracy" and col != "FastICA Accuracy"]
    print(f"  Discovered accuracy columns: {discovered_accuracy_cols}")

    if not discovered_accuracy_cols:
        print(f"  WARNING: No 'Accuracy' columns found. Skipping this dataset plot.")
        ax.text(0.5, 0.5, "No accuracy data found", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(No accuracy data)", fontsize=16)
        continue

    current_plot_attributes = []
    print("  Processing attributes for each method:")
    for col_name in discovered_accuracy_cols:
        # Clean base_method_name: strip whitespace from col_name, remove " Accuracy", then strip again.
        base_method_name = col_name.strip().replace(" Accuracy", "").strip()

        # 1. Determine color
        assigned_color = None
        # Check if full column name (col_name) is in method_styles and has a color
        if col_name in method_styles and 'color' in method_styles[col_name]:
            assigned_color = method_styles[col_name]['color']

        if assigned_color is None: # Not in method_styles by col_name or no color specified there
            # Check if base_method_name has an already assigned global color (for new methods)
            if base_method_name in globally_assigned_new_method_colors:
                assigned_color = globally_assigned_new_method_colors[base_method_name]
            else:
                # Assign a new color from the cycle
                if global_color_cycler_index < len(unique_default_colors):
                    assigned_color = unique_default_colors[global_color_cycler_index]
                else: # Fallback if unique_default_colors are exhausted
                    assigned_color = default_color_palette[(global_color_cycler_index - len(unique_default_colors)) % len(default_color_palette)]
                globally_assigned_new_method_colors[base_method_name] = assigned_color
                global_color_cycler_index += 1

        # 2. Determine X-axis label
        # Default x-axis display name: from method_styles short_name (using col_name key) or base_method_name
        default_xaxis_display_name = base_method_name # Fallback is the cleaned base_method_name
        if col_name in method_styles and 'short_name' in method_styles[col_name]:
            default_xaxis_display_name = method_styles[col_name]['short_name']

        # Override with specific abbreviation if base_method_name is in xaxis_abbreviations
        x_axis_label = xaxis_abbreviations.get(base_method_name, default_xaxis_display_name)

        # --- DEBUG PRINT for specific methods ---
        if base_method_name in debug_methods_for_xaxis:
            print(f"    - Method: '{base_method_name}' (from col: '{col_name}')")
            print(f"      Is key in xaxis_abbreviations? {base_method_name in xaxis_abbreviations}")
            print(f"      Default X-axis name: '{default_xaxis_display_name}'")
            print(f"      Final X-axis label: '{x_axis_label}'")
        # --- END DEBUG ---

        current_plot_attributes.append({
            'full_name': col_name,
            'x_axis_label': x_axis_label,
            'color': assigned_color,
            'legend_label': base_method_name
        })

        if base_method_name not in final_legend_info:
            final_legend_info[base_method_name] = assigned_color

    accuracy_cols_to_use = [attr['full_name'] for attr in current_plot_attributes]
    bar_colors_for_plot = [attr['color'] for attr in current_plot_attributes]
    x_tick_labels_for_plot = ['MPAD', 'UMAP', 'Isomap', 'KPCA', 'MDS', 'AE', 'FeatAgg', 'LLE', 'NMF', 'RanProj', 'VAE', 'tSNE']
    print(f"  X-tick labels for this plot: {x_tick_labels_for_plot}")

    # ... (rest of your data processing and plotting logic for the subplot)
    try:
        grouped_param_means = df.groupby(['alpha', 'b'])[accuracy_cols_to_use].mean()
    except KeyError as e:
        print(f"  ERROR: Column {e} not found for grouping in {csv_path}. Skipping.")
        ax.text(0.5, 0.5, f"Column {e} not found", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Column {e} not found)", fontsize=16)
        continue
    if grouped_param_means.empty:
        print(f"  WARNING: No data after grouping for {csv_path}. Skipping plot.")
        ax.text(0.5, 0.5, "No data after grouping", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(No data after grouping)", fontsize=16)
        continue

    selection_metric_series = None
    mpad_full_name = next((name for name in accuracy_cols_to_use if name.strip().upper().startswith("MPAD")), None)

    if mpad_full_name:
        def calculate_mpad_centric_score(row_series):
            mpad_accuracy = row_series[mpad_full_name]
            other_method_accuracies = [row_series[col] for col in accuracy_cols_to_use if col != mpad_full_name]
            if not other_method_accuracies: return mpad_accuracy
            return mpad_accuracy - max(other_method_accuracies) if other_method_accuracies else mpad_accuracy
        selection_metric_series = grouped_param_means.apply(calculate_mpad_centric_score, axis=1)
    elif accuracy_cols_to_use:
        selection_metric_series = grouped_param_means[accuracy_cols_to_use].max(axis=1)
    else:
        print(f"  WARNING: No accuracy columns to determine selection metric for {csv_path}. Skipping plot.")
        ax.text(0.5, 0.5, "No methods for metric", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(No methods for metric)", fontsize=16)
        continue

    if selection_metric_series.empty:
        print(f"  WARNING: Selection metric series is empty for {csv_path}. Skipping plot.")
        ax.text(0.5, 0.5, "Metric calculation failed", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Metric calculation failed)", fontsize=16)
        continue

    best_params_combination_idx = selection_metric_series.idxmax()
    selected_alpha_val, selected_b_val = best_params_combination_idx

    mask_for_best_params = (df['alpha'] == selected_alpha_val) & (df['b'] == selected_b_val)
    data_at_best_params = df[mask_for_best_params]

    if data_at_best_params.empty:
        mean_abs_accuracies = grouped_param_means.loc[best_params_combination_idx]
    else:
        mean_abs_accuracies = data_at_best_params[accuracy_cols_to_use].mean()

    if mean_abs_accuracies.isnull().all():
        ax.text(0.5, 0.5, "Mean accuracies are NaN", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Mean accuracies NaN)", fontsize=16)
        continue

    max_absolute_accuracy = mean_abs_accuracies.max()
    if pd.isna(max_absolute_accuracy) or max_absolute_accuracy == 0:
        relative_mean_accuracies_series = mean_abs_accuracies.fillna(0)
    else:
        relative_mean_accuracies_series = mean_abs_accuracies / max_absolute_accuracy

    ordered_relative_accuracies = [relative_mean_accuracies_series[full_name] for full_name in accuracy_cols_to_use]

    best_performing_method_fullname = mean_abs_accuracies.idxmax() if not mean_abs_accuracies.isnull().all() else None

    x_indices = np.arange(len(accuracy_cols_to_use))
    bar_width = 0.6
    bars = ax.bar(x_indices, ordered_relative_accuracies, bar_width, color=bar_colors_for_plot)

    if best_performing_method_fullname and best_performing_method_fullname in accuracy_cols_to_use:
        try:
            idx_of_baseline_method = accuracy_cols_to_use.index(best_performing_method_fullname)
            bars[idx_of_baseline_method].set_edgecolor('black')
            bars[idx_of_baseline_method].set_linewidth(2.5)
        except ValueError:
            pass # Should not happen if logic is correct

    ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5)
    current_max_rel_acc_val = np.nanmax(ordered_relative_accuracies) if ordered_relative_accuracies else 1.0
    ax.set_ylim(0, max(1.15, current_max_rel_acc_val * 1.05 if pd.notna(current_max_rel_acc_val) else 1.15) )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_tick_labels_for_plot, rotation=0, ha='center', fontsize=14)
    ax.set_ylabel("Relative Average Accuracy", fontsize=16)

    baseline_text = f"Selected: $\\alpha = {selected_alpha_val:.2g}$, $b = {selected_b_val}$"
    ax.text(0.5, -0.25, baseline_text, transform=ax.transAxes, ha='center', va='center', fontsize=14)

# -------------------------------
# 5. Create Global Legend and Final Adjustments
# -------------------------------
print("\n--- Generating Legend and Final Plot ---")
if final_legend_info:
    legend_labels = list(final_legend_info.keys())
    legend_handles = [plt.Rectangle((0,0),1,1, color=final_legend_info[label]) for label in legend_labels]
    print(f"Legend labels: {legend_labels}")
    print(f"Legend colors: {[final_legend_info[label] for label in legend_labels]}")

    num_legend_cols = (len(legend_labels) + 2) // 3
    if len(legend_labels) <= 4: num_legend_cols = (len(legend_labels) +1 ) // 2
    elif len(legend_labels) <= 6: num_legend_cols = 3
    else: num_legend_cols = 6

    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=num_legend_cols,
               bbox_to_anchor=(0.5, 0.945), title="Baselines", fontsize=14, title_fontsize=16)

# plt.suptitle("Relative Performance of Dimensionality Reduction Methods", fontsize=24, y=0.995)
plt.subplots_adjust(top=0.88, bottom=0.15, left=0.07, right=0.98, hspace=0.55, wspace=0.25)
plt.show()
print("--- Plotting Complete ---")