import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 24})
plt.style.use('tableau-colorblind10')

# ---------------------------------------------------
# 1. Define CSV paths and dataset names.
# ---------------------------------------------------
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

# ---------------------------------------------------
# 2. Define the FIXED X-AXIS methods, their labels, and mapping to CSV column names.
# ---------------------------------------------------
display_methods_config = [
    {'x_label': 'MPAD',    'full_csv_col_name': 'MPAD Accuracy'},
    {'x_label': 'UMAP',    'full_csv_col_name': 'UMAP Accuracy'},
    {'x_label': 'Isomap',  'full_csv_col_name': 'Isomap Accuracy'},
    {'x_label': 'KPCA',    'full_csv_col_name': 'KernelPCA Accuracy'},
    {'x_label': 'AE',      'full_csv_col_name': 'Autoencoder Accuracy'},
    {'x_label': 'FeatAgg', 'full_csv_col_name': 'Feature Agglomeration Accuracy'},
    {'x_label': 'LLE',     'full_csv_col_name': 'LLE Accuracy'},
    {'x_label': 'NMF',     'full_csv_col_name': 'NMF Accuracy'},
    {'x_label': 'RandProj', 'full_csv_col_name': 'Random Projection Accuracy'},
    {'x_label': 'VAE',     'full_csv_col_name': 'VAE Accuracy'},
    {'x_label': 'tSNE',    'full_csv_col_name': 't-SNE Accuracy'},
    {'x_label': 'LSH',    'full_csv_col_name': 'LSH Accuracy'}
]

predefined_method_attributes = {
    'MPAD Accuracy':     {'color': 'red'},
    'UMAP Accuracy':     {'color': 'green'},
    'Isomap Accuracy':   {'color': 'orange'},
    'KernelPCA Accuracy':{'color': 'plum'},
    'Autoencoder Accuracy': {'color': 'blue'},
    'Feature Agglomeration Accuracy': {'color': 'blue'},
    'LLE Accuracy': {'color': 'gray'},
    'NMF Accuracy': {'color': 'yellowgreen'},
    'Random Projection Accuracy': {'color': 'brown'},
    'VAE Accuracy': {'color': 'tan'},
    't-SNE Accuracy': {'color': 'teal'},
    'LSH Accuracy': {'color': 'olive'}
}
default_color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
known_colors = {style['color'] for style in predefined_method_attributes.values() if 'color' in style}
unique_default_colors = [color for color in default_color_palette if color not in known_colors]
globally_assigned_new_method_styles = {}
global_color_cycler_index = 0

styled_display_methods = []
for method_config in display_methods_config:
    full_name = method_config['full_csv_col_name']
    x_label = method_config['x_label']
    color = None
    if full_name in predefined_method_attributes and 'color' in predefined_method_attributes[full_name]:
        color = predefined_method_attributes[full_name]['color']
    elif full_name in globally_assigned_new_method_styles and 'color' in globally_assigned_new_method_styles[full_name]:
        color = globally_assigned_new_method_styles[full_name]['color']
    else:
        if global_color_cycler_index < len(unique_default_colors):
            color = unique_default_colors[global_color_cycler_index]
        else:
            color = default_color_palette[(global_color_cycler_index - len(unique_default_colors)) % len(default_color_palette)]
        global_color_cycler_index += 1
        globally_assigned_new_method_styles[full_name] = {'color': color}
    styled_display_methods.append({
        'full_csv_col_name': full_name,
        'x_label': x_label,
        'color': color
    })

fixed_plot_x_tick_labels = [item['x_label'] for item in styled_display_methods]
fixed_plot_bar_colors = [item['color'] for item in styled_display_methods]
fixed_plot_method_fullnames = [item['full_csv_col_name'] for item in styled_display_methods]

EXCLUDED_METHODS = ["PCA Accuracy", "FastICA Accuracy", "MDS Accuracy"]

# ---------------------------------------------------
# 3. Create a 2x2 subplot layout.
# ---------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(17, 14))
axs = axs.flatten()

# Initialize variable to store the overall maximum y_value for counts
overall_max_y_count = 0

# ---------------------------------------------------
# 4. Process each dataset.
# ---------------------------------------------------
for i, csv_path in enumerate(csv_paths):
    ax = axs[i]
    current_max_y_for_this_plot = 0 # To store max_y for the current plot before updating global
    try:
        df = pd.read_csv(csv_path)
        if 'b' in df.columns:
            df = df[~df['b'].isin([40, 50])]
        else:
            print(f"Warning: Column 'b' not found in {csv_path}. Skipping filtering by 'b'.")

    except FileNotFoundError:
        print(f"Warning: CSV file not found at {csv_path}. Skipping.")
        ax.text(0.5, 0.5, "CSV not found", ha='center', va='center', fontsize=14)
        ax.set_title(dataset_names[i], fontsize=24)
        ax.set_xticks([])
        ax.set_yticks([])
        # We will set a common Y axis later, so just continue
        continue
    except Exception as e:
        print(f"Warning: Error loading {csv_path}: {e}. Skipping.")
        ax.text(0.5, 0.5, "Error loading data", ha='center', va='center', fontsize=14)
        ax.set_title(dataset_names[i], fontsize=24)
        ax.set_xticks([])
        ax.set_yticks([])
        # We will set a common Y axis later
        continue

    best_counts = {full_name: 0 for full_name in fixed_plot_method_fullnames}
    second_counts = {full_name: 0 for full_name in fixed_plot_method_fullnames}

    actual_methods_in_csv_header = [
        col for col in df.columns if col.endswith("Accuracy") and col not in EXCLUDED_METHODS
    ]

    ax.set_title(dataset_names[i], fontsize=24) # Set title early

    if not actual_methods_in_csv_header:
        print(f"No relevant (and non-excluded) accuracy columns in {csv_path}")
        # Still set up basic plot structure; Y axis will be harmonized later
        ax.set_xticks(np.arange(len(fixed_plot_x_tick_labels)))
        ax.set_xticklabels(fixed_plot_x_tick_labels, rotation=45, ha='right', fontsize=20)
        ax.set_ylabel("Count", fontsize=24)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.bar(np.arange(len(fixed_plot_method_fullnames)), np.zeros(len(fixed_plot_method_fullnames)),
               width=0.6, color=fixed_plot_bar_colors)
        ax.bar(np.arange(len(fixed_plot_method_fullnames)), np.zeros(len(fixed_plot_method_fullnames)),
               bottom=np.zeros(len(fixed_plot_method_fullnames)), width=0.6, color='lightgray', alpha=0.7)
        current_max_y_for_this_plot = 0 # No data, so max count is 0 for this plot
    else:
        for _, row in df.iterrows():
            available_methods_in_row = [
                m for m in actual_methods_in_csv_header if m in row.index and pd.notna(row[m])
            ]
            if not available_methods_in_row:
                continue

            accuracies = row[available_methods_in_row]
            if len(accuracies.dropna()) < 1:
                continue

            ranks = accuracies.rank(method='dense', ascending=False)

            best_rank_val = ranks.min()
            current_row_best_methods = [m for m in available_methods_in_row if ranks.get(m, float('inf')) == best_rank_val]
            for best_m_fullname in current_row_best_methods:
                if best_m_fullname in best_counts:
                    best_counts[best_m_fullname] += 1

            if len(ranks.unique()) > 1:
                sorted_unique_ranks = sorted(ranks.dropna().unique())
                if len(sorted_unique_ranks) > 1:
                    second_best_rank_val = sorted_unique_ranks[1]
                    current_row_s_best_methods = [m for m in available_methods_in_row if ranks.get(m, float('inf')) == second_best_rank_val]
                    for second_m_fullname in current_row_s_best_methods:
                        if second_m_fullname in second_counts:
                            second_counts[second_m_fullname] += 1

        counts_df = pd.DataFrame({
            'Method_FullName': fixed_plot_method_fullnames,
            'Best Count': [best_counts.get(m_full, 0) for m_full in fixed_plot_method_fullnames],
            'Second Best Count': [second_counts.get(m_full, 0) for m_full in fixed_plot_method_fullnames]
        })

        x_indices = np.arange(len(fixed_plot_method_fullnames))
        width = 0.5

        bars_best = ax.bar(x_indices, counts_df['Best Count'], width, color=fixed_plot_bar_colors, label="Best Count")
        bars_second = ax.bar(x_indices, counts_df['Second Best Count'], width,
                             bottom=counts_df['Best Count'], color='lightgray', alpha=0.7, label="Second Best Count")

        mpad_full_name_target = 'MPAD Accuracy'
        if mpad_full_name_target in fixed_plot_method_fullnames:
            try:
                mpad_idx = fixed_plot_method_fullnames.index(mpad_full_name_target)
                if mpad_idx < len(bars_best) and counts_df.loc[counts_df['Method_FullName'] == mpad_full_name_target, 'Best Count'].iloc[0] > 0:
                    bars_best[mpad_idx].set_edgecolor('black')
                    bars_best[mpad_idx].set_linewidth(2.5)
                if mpad_idx < len(bars_second) and counts_df.loc[counts_df['Method_FullName'] == mpad_full_name_target, 'Second Best Count'].iloc[0] > 0:
                    bars_second[mpad_idx].set_edgecolor('black')
                    bars_second[mpad_idx].set_linewidth(1.5)
                    bars_second[mpad_idx].set_linestyle('--')
            except (ValueError, IndexError):
                pass

        ax.set_xticks(x_indices)
        ax.set_xticklabels(fixed_plot_x_tick_labels, rotation=45, ha='right', fontsize=20)
        ax.set_ylabel("Count", fontsize=24)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        current_max_y_for_this_plot = (counts_df['Best Count'] + counts_df['Second Best Count']).max()

    # Update the overall_max_y_count
    if pd.notna(current_max_y_for_this_plot) and current_max_y_for_this_plot > overall_max_y_count:
        overall_max_y_count = current_max_y_for_this_plot

# ---------------------------------------------------
# 5. Apply uniform Y-axis to all subplots and create global legend.
# ---------------------------------------------------
final_y_limit = overall_max_y_count * 1.15 if overall_max_y_count > 0 else 10
for ax_item in axs:
    ax_item.set_ylim(0, final_y_limit)
    # Ensure ylabel and tick_params are set even for error plots (if not already)
    if not ax_item.get_ylabel(): # Set if not already set (e.g. for error plots)
        ax_item.set_ylabel("Count", fontsize=24)
    ax_item.tick_params(axis='y', labelsize=14) # Ensure consistent y-tick label size
    # Ensure grid for consistency
    ax_item.grid(axis='y', linestyle='--', alpha=0.7, which='major')


legend_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=12, markeredgewidth=1.0, markeredgecolor='black',
           label='Best Count (color by method)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', alpha=0.7, markersize=12, markeredgecolor='black',
           label='Second Best Count'),
]
mpad_style_for_legend = predefined_method_attributes.get('MPAD Accuracy', {})
mpad_color_for_legend = mpad_style_for_legend.get('color', 'red')

legend_handles.append(
    Line2D([0], [0], marker='s', color='w', markerfacecolor=mpad_color_for_legend,
           markeredgecolor='black', markeredgewidth=2.5, markersize=12,
           label='MPAD (Count Emphasized)')
)

plt.tight_layout(rect=[0, 0.05, 1, 0.90])
fig.legend(handles=legend_handles, loc='upper center', ncol=3, fontsize=18, bbox_to_anchor=(0.5, 0.97))

plt.show()