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
#    USER MUST VERIFY 'full_csv_col_name' values match their CSV headers EXACTLY.
# ---------------------------------------------------
display_methods_config = [
    {'x_label': 'MPAD',    'full_csv_col_name': 'MPAD Accuracy'},
    {'x_label': 'UMAP',    'full_csv_col_name': 'UMAP Accuracy'},
    {'x_label': 'Isomap',  'full_csv_col_name': 'Isomap Accuracy'},
    {'x_label': 'KPCA',    'full_csv_col_name': 'KernelPCA Accuracy'}, # Verify: KernelPCA or KPCA in CSV?
    {'x_label': 'MDS',     'full_csv_col_name': 'MDS Accuracy'},
    {'x_label': 'AE',      'full_csv_col_name': 'Autoencoder Accuracy'}, # Verify full name
    {'x_label': 'FeatAgg', 'full_csv_col_name': 'Feature Agglomeration Accuracy'}, # Verify full name
    {'x_label': 'LLE',     'full_csv_col_name': 'LLE Accuracy'}, # Verify full name
    {'x_label': 'NMF',     'full_csv_col_name': 'NMF Accuracy'}, # Verify full name
    {'x_label': 'RandProj', 'full_csv_col_name': 'Random Projection Accuracy'}, # Verify full name
    {'x_label': 'VAE',     'full_csv_col_name': 'VAE Accuracy'}, # Verify full name
    {'x_label': 'tSNE',    'full_csv_col_name': 't-SNE Accuracy'} # Verify full name (e.g. t-SNE or TSNE)
]

# Predefined attributes (primarily for color) for known methods.
# Keys should be the 'full_csv_col_name'.
predefined_method_attributes = {
    'MPAD Accuracy':     {'color': 'red'},
    'UMAP Accuracy':     {'color': 'green'},
    'Isomap Accuracy':   {'color': 'orange'},
    'KernelPCA Accuracy':{'color': 'purple'},
    'MDS Accuracy':      {'color': 'gray'},
    # Add other methods from display_methods_config here if they have preferred colors:
    'Autoencoder Accuracy': {'color': 'blue'},
    'Feature Agglomeration Accuracy': {'color': 'cyan'},
    'LLE Accuracy': {'color': 'magenta'},
    'NMF Accuracy': {'color': 'lime'}, # Example color
    'Random Projection Accuracy': {'color': 'brown'},
    'VAE Accuracy': {'color': 'pink'},
    't-SNE Accuracy': {'color': 'teal'},
}

# Color cycling setup
default_color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
known_colors = {style['color'] for style in predefined_method_attributes.values() if 'color' in style}
unique_default_colors = [color for color in default_color_palette if color not in known_colors]
# Stores assigned styles for methods not in predefined_method_attributes, keyed by full_csv_col_name
globally_assigned_new_method_styles = {}
global_color_cycler_index = 0

# --- Style the fixed list of display methods ---
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

# Extract attributes for plotting in the fixed order
fixed_plot_x_tick_labels = [item['x_label'] for item in styled_display_methods]
fixed_plot_bar_colors = [item['color'] for item in styled_display_methods]
# This is the ordered list of full method names we expect on the plot
fixed_plot_method_fullnames = [item['full_csv_col_name'] for item in styled_display_methods]

# ---------------------------------------------------
# 3. Create a 2x2 subplot layout.
# ---------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(15, 12)) # Adjusted for potentially more x-labels
axs = axs.flatten()

# ---------------------------------------------------
# 4. Process each dataset.
# ---------------------------------------------------
for i, csv_path in enumerate(csv_paths):
    ax = axs[i]
    try:
        df = pd.read_csv(csv_path)
        df = df[~df['b'].isin([40, 50])]
    except FileNotFoundError:
        print(f"Warning: CSV file not found at {csv_path}. Skipping.")
        ax.text(0.5, 0.5, "CSV not found", ha='center', va='center')
        ax.set_title(dataset_names[i], fontsize=24)
        ax.set_xticks([]) # Clear ticks for empty plot
        ax.set_yticks([])
        continue
    except Exception as e:
        print(f"Warning: Error loading {csv_path}: {e}. Skipping.")
        ax.text(0.5, 0.5, "Error loading data", ha='center', va='center')
        ax.set_title(dataset_names[i], fontsize=24)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    # Initialize counters for the fixed set of display methods
    best_counts = {full_name: 0 for full_name in fixed_plot_method_fullnames}
    second_counts = {full_name: 0 for full_name in fixed_plot_method_fullnames}

    # Discover methods ACTUALLY PRESENT in this CSV's header for data access
    actual_methods_in_csv_header = [
        col for col in df.columns if col.endswith("Accuracy") and col not in ["PCA Accuracy", "FastICA Accuracy"]
    ]
    if not actual_methods_in_csv_header:
        print(f"No relevant accuracy columns in {csv_path}")
        # Setup empty plot appearance for consistency
        ax.set_title(dataset_names[i], fontsize=24)
        ax.set_xticks(np.arange(len(fixed_plot_x_tick_labels)))
        ax.set_xticklabels(fixed_plot_x_tick_labels, rotation=45, ha='right', fontsize=24)
        ax.set_ylabel("Count", fontsize=24)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Plot zero bars
        ax.bar(np.arange(len(fixed_plot_method_fullnames)), np.zeros(len(fixed_plot_method_fullnames)),
               width=0.6, color=fixed_plot_bar_colors)
        ax.bar(np.arange(len(fixed_plot_method_fullnames)), np.zeros(len(fixed_plot_method_fullnames)),
               bottom=np.zeros(len(fixed_plot_method_fullnames)), width=0.6, color='lightgray', alpha=0.7)

        continue


    # Loop over each row, rank methods, count best/second best
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
        current_row_best_methods = [m for m in available_methods_in_row if ranks[m] == best_rank_val]
        for best_m_fullname in current_row_best_methods:
            if best_m_fullname in best_counts: # Only count if it's one of the 12 methods we display
                best_counts[best_m_fullname] += 1

        if len(ranks.unique()) > 1:
            sorted_unique_ranks = sorted(ranks.unique())
            if len(sorted_unique_ranks) > 1:
                second_best_rank_val = sorted_unique_ranks[1]
                current_row_s_best_methods = [m for m in available_methods_in_row if ranks[m] == second_best_rank_val]
                for second_m_fullname in current_row_s_best_methods:
                    if second_m_fullname in second_counts: # Only count if we display it
                        second_counts[second_m_fullname] += 1

    # Create a DataFrame for plotting, ensuring order matches fixed_plot_method_fullnames
    counts_df = pd.DataFrame({
        'Method_FullName': fixed_plot_method_fullnames,
        'Best Count': [best_counts.get(m_full, 0) for m_full in fixed_plot_method_fullnames],
        'Second Best Count': [second_counts.get(m_full, 0) for m_full in fixed_plot_method_fullnames]
    })

    # Plot a stacked bar chart
    x_indices = np.arange(len(fixed_plot_method_fullnames)) # Should be 12
    width = 0.5 # Adjusted width for more bars

    bars_best = ax.bar(x_indices, counts_df['Best Count'], width, color=fixed_plot_bar_colors)
    bars_second = ax.bar(x_indices, counts_df['Second Best Count'], width,
                         bottom=counts_df['Best Count'], color='lightgray', alpha=0.7)

    mpad_full_name_target = 'MPAD Accuracy'
    if mpad_full_name_target in fixed_plot_method_fullnames:
        try:
            mpad_idx = fixed_plot_method_fullnames.index(mpad_full_name_target)
            if mpad_idx < len(bars_best): # Ensure index is valid
                bars_best[mpad_idx].set_edgecolor('black')
                bars_best[mpad_idx].set_linewidth(2.5)
        except ValueError:
            pass

    ax.set_xticks(x_indices)
    ax.set_xticklabels(fixed_plot_x_tick_labels, rotation=45, ha='right', fontsize=24)
    ax.set_ylabel("Count", fontsize=24)
    ax.set_title(dataset_names[i], fontsize=24)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Ensure y-limit accommodates counts
    max_y = (counts_df['Best Count'] + counts_df['Second Best Count']).max()
    ax.set_ylim(0, max_y * 1.1 if max_y > 0 else 10) # Set a default if all counts are 0


# ---------------------------------------------------
# 5. Create one global legend.
# ---------------------------------------------------
legend_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
           markeredgecolor='black', markersize=12, markeredgewidth=1.5,
           label='Best Count (bar color by method)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', alpha=0.7, markersize=12,
           label='Second Best Count'),
]
# Add MPAD emphasis to legend only if MPAD is part of the display and has a style
mpad_style_for_legend = predefined_method_attributes.get('MPAD Accuracy', {})
mpad_color_for_legend = mpad_style_for_legend.get('color', 'red') # Default to red if not found

legend_handles.append(
    Line2D([0], [0], marker='s', color='w', markerfacecolor=mpad_color_for_legend,
           markeredgecolor='black', markeredgewidth=2.5, markersize=12,
           label='MPAD (Best Count portion emphasized)')
)


plt.tight_layout(rect=[0, 0.05, 1, 0.90])
fig.legend(handles=legend_handles, loc='upper center', ncol=1, fontsize=18, bbox_to_anchor=(0.5, 0.99))

plt.show()