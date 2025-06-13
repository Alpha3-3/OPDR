import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Apply global style and font size
plt.style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 18})

# -------------------------------
# 1. Define display config and predefined attributes
# -------------------------------
display_methods_config = [
    {'x_label': 'MPAD',    'full_csv_col_name': 'MPAD Accuracy'},
    {'x_label': 'UMAP',    'full_csv_col_name': 'UMAP Accuracy'},
    {'x_label': 'Isomap',  'full_csv_col_name': 'Isomap Accuracy'},
    {'x_label': 'KPCA',    'full_csv_col_name': 'KernelPCA Accuracy'},
    {'x_label': 'AE',      'full_csv_col_name': 'Autoencoder Accuracy'},
    {'x_label': 'FA', 'full_csv_col_name': 'FeatureAgglomeration Accuracy'},
    {'x_label': 'LLE',     'full_csv_col_name': 'LLE Accuracy'},
    {'x_label': 'NMF',     'full_csv_col_name': 'NMF Accuracy'},
    {'x_label': 'RP','full_csv_col_name': 'RandomProjection Accuracy'},
    {'x_label': 'VAE',     'full_csv_col_name': 'VAE Accuracy'},
    {'x_label': 'tSNE',    'full_csv_col_name': 'tSNE Accuracy'},
    {'x_label': 'LSH',     'full_csv_col_name': 'LSH Accuracy'}
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


# Build method_styles and xaxis_abbreviations dynamically
method_styles = {}
xaxis_abbreviations = {}
for conf in display_methods_config:
    full_name = conf['full_csv_col_name']
    base_name = full_name.replace(' Accuracy', '').strip()
    attrs = {
        'short_name': conf['x_label']
    }
    # assign predefined color if available
    if full_name in predefined_method_attributes:
        attrs['color'] = predefined_method_attributes[full_name]['color']
    method_styles[full_name] = attrs
    xaxis_abbreviations[base_name] = conf['x_label']

# Methods for debugging x-axis label generation
debug_methods_for_xaxis = set(xaxis_abbreviations.keys())

# -------------------------------
# 2. Prepare color cycling
# -------------------------------
default_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
predefined_colors = {v['color'] for v in method_styles.values() if 'color' in v}
unique_default_colors = [c for c in default_palette if c not in predefined_colors]

globally_assigned_new_method_colors = {}
global_color_cycler_index = 0

final_legend_info = {}

# -------------------------------
# 3. Create 2x2 subplot layout
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(17, 13))
axs = axs.flatten()

# -------------------------------
# 4. Process each dataset and plot
# -------------------------------
print("--- Starting Dataset Processing ---")
csv_paths = [
    "parameter_sweep_results_Fasttext_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_Isolet_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_Arcene_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_PBMC3k_Multiple_methods_with_additional_baselines.csv"
]
dataset_names = ["Fasttext", "Isolet", "Arcene", "PBMC3k"]

for i, path in enumerate(csv_paths):
    ax = axs[i]
    ax.set_title(dataset_names[i], fontsize=20)
    print(f"\nProcessing Dataset: {dataset_names[i]} ({path})")

    try:
        df = pd.read_csv(path)
        df = df[~df['b'].isin([40, 50])]
    except FileNotFoundError:
        ax.text(0.5, 0.5, "CSV file not found", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Data not found)", fontsize=16)
        continue
    except Exception as e:
        ax.text(0.5, 0.5, "Error loading CSV", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Error loading data)", fontsize=16)
        continue

    cols = [c for c in df.columns if c.endswith('Accuracy') and c not in ['PCA Accuracy', 'FastICA Accuracy', 'MDS Accuracy']]
    print(f"  Discovered accuracy columns: {cols}")
    if not cols:
        ax.text(0.5, 0.5, "No accuracy data found", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(No accuracy data)", fontsize=16)
        continue

    plot_attrs = []
    print("  Processing attributes for each method:")
    for col in cols:
        base = col.replace(' Accuracy', '').strip()
        # 1. Determine color
        color = method_styles.get(col, {}).get('color')
        if color is None:
            if base in globally_assigned_new_method_colors:
                color = globally_assigned_new_method_colors[base]
            else:
                if global_color_cycler_index < len(unique_default_colors):
                    color = unique_default_colors[global_color_cycler_index]
                else:
                    color = default_palette[(global_color_cycler_index - len(unique_default_colors)) % len(default_palette)]
                globally_assigned_new_method_colors[base] = color
                global_color_cycler_index += 1

        # 2. Determine x-axis label
        default_label = method_styles.get(col, {}).get('short_name', base)
        x_label = xaxis_abbreviations.get(base, default_label)

        if base in debug_methods_for_xaxis:
            print(f"    - Method: '{base}' (col: '{col}') | X-axis: '{x_label}' | Color: '{color}'")

        plot_attrs.append({'full_name': col, 'x_axis_label': x_label, 'color': color, 'legend_label': base})
        final_legend_info[base] = color

    accuracy_cols = [a['full_name'] for a in plot_attrs]
    colors = [a['color'] for a in plot_attrs]
    x_labels = [a['x_axis_label'] for a in plot_attrs]
    print(f"  X-tick labels for this plot: {x_labels}")

    # Group, select best params, compute mean accuracies, normalize, and plot (unchanged logic)
    try:
        grouped = df.groupby(['alpha', 'b'])[accuracy_cols].mean()
    except KeyError as e:
        ax.text(0.5, 0.5, f"Column {e} not found", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Column {e} not found)", fontsize=16)
        continue
    if grouped.empty:
        ax.text(0.5, 0.5, "No data after grouping", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(No data after grouping)", fontsize=16)
        continue

    mpad_full = next((n for n in accuracy_cols if n.upper().startswith('MPAD')), None)
    if mpad_full:
        def score(row):
            mpad_val = row[mpad_full]
            others = [row[c] for c in accuracy_cols if c != mpad_full]
            return mpad_val - max(others) if others else mpad_val
        metric = grouped.apply(score, axis=1)
    else:
        metric = grouped[accuracy_cols].max(axis=1)
    if metric.empty:
        ax.text(0.5, 0.5, "Metric calculation failed", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Metric calculation failed)", fontsize=16)
        continue

    best_idx = metric.idxmax()
    alpha_val, b_val = best_idx
    subset = df[(df['alpha']==alpha_val)&(df['b']==b_val)]
    if subset.empty:
        mean_acc = grouped.loc[best_idx]
    else:
        mean_acc = subset[accuracy_cols].mean()
    if mean_acc.isnull().all():
        ax.text(0.5, 0.5, "Mean accuracies are NaN", ha='center', va='center', fontsize=14)
        ax.set_title(f"{dataset_names[i]}\n(Mean accuracies NaN)", fontsize=16)
        continue

    max_acc = mean_acc.max()
    rel_acc = mean_acc.fillna(0) if max_acc in [0, np.nan] else mean_acc / max_acc
    ordered = [rel_acc[col] for col in accuracy_cols]
    bars = ax.bar(np.arange(len(accuracy_cols)), ordered, 0.6, color=colors)

    if mean_acc.idxmax() in accuracy_cols:
        idx = accuracy_cols.index(mean_acc.idxmax())
        bars[idx].set_edgecolor('black')
        bars[idx].set_linewidth(2.5)

    ax.axhline(1.0, color='black', linestyle=':', linewidth=1.5)
    current_max = np.nanmax(ordered) if ordered else 1.0
    ax.set_ylim(0, max(1.15, current_max*1.05))

    ax.set_xticks(np.arange(len(accuracy_cols)))
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=14)
    ax.set_ylabel("Relative Average Accuracy", fontsize=16)
    ax.text(0.5, -0.25, f"Selected: $\\alpha={alpha_val:.2g}$, $b={b_val}$", transform=ax.transAxes, ha='center', va='center', fontsize=14)

# -------------------------------
# 5. Global legend and adjustments
# -------------------------------
print("\n--- Generating Legend and Final Plot ---")
if final_legend_info:
    labels = list(final_legend_info.keys())
    handles = [plt.Rectangle((0,0),1,1, color=final_legend_info[l]) for l in labels]
    num_cols = (len(labels)+2)//3
    if len(labels)<=4: num_cols = (len(labels)+1)//2
    elif len(labels)<=6: num_cols = 3
    else: num_cols = 6
    fig.legend(handles, labels, loc='upper center', ncol=num_cols,
               bbox_to_anchor=(0.5,1), title="Baselines", fontsize=14, title_fontsize=16)

plt.subplots_adjust(top=0.88, bottom=0.15, left=0.07, right=0.98, hspace=0.55, wspace=0.25)
plt.show()
print("--- Plotting Complete ---")
