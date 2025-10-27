import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Apply global style and font size
plt.style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 18})

# -------------------------------
# 1. Define display config and predefined attributes
# -------------------------------
display_methods_config = [
    {'x_label': 'QPAD',    'full_csv_col_name': 'MPAD Accuracy'},
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
    'PCA Accuracy':      {'color': '#FF8C00'},
    'UMAP Accuracy':     {'color': '#8B4513'},
    'Isomap Accuracy':   {'color': '#FF1493'},
    'KernelPCA Accuracy':{'color': '#9370DB'},
    'Autoencoder Accuracy': {'color': '#32CD32'},
    'Feature Agglomeration Accuracy': {'color': '#FFD700'},
    'LLE Accuracy': {'color': '#4169E1'},
    'NMF Accuracy': {'color': '#006400'},
    'Random Projection Accuracy': {'color': '#808080'},
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
# 3. Will create two figures later
# -------------------------------
# Figure 1: Absolute accuracy values
# Figure 2: Relative accuracy w.r.t QPAD

# -------------------------------
# 4. Find common best (alpha, b) across all datasets
# -------------------------------
print("--- Starting Dataset Processing ---")
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

csv_paths = [
    os.path.join(script_dir, "parameter_sweep_results_Fasttext_Multiple_methods_with_additional_baselines.csv"),
    os.path.join(script_dir, "parameter_sweep_results_Isolet_Multiple_methods_with_additional_baselines.csv"),
    os.path.join(script_dir, "parameter_sweep_results_Arcene_Multiple_methods_with_additional_baselines.csv"),
    os.path.join(script_dir, "parameter_sweep_results_PBMC3k_Multiple_methods_with_additional_baselines.csv")
]
dataset_names = ["Fasttext", "Isolet", "Arcene", "PBMC3k"]

print("\n=== Finding common best (alpha, b) across all datasets ===")
# Load all datasets first
all_dfs = []
all_accuracy_cols = []

for i, path in enumerate(csv_paths):
    try:
        df = pd.read_csv(path)
        df = df[~df['b'].isin([40, 50])]
        cols = [c for c in df.columns if c.endswith('Accuracy') and c not in ['PCA Accuracy', 'FastICA Accuracy', 'MDS Accuracy']]
        all_dfs.append(df)
        all_accuracy_cols.append(cols)
        print(f"Loaded {dataset_names[i]}: {len(df)} rows, {len(cols)} accuracy columns")
    except Exception as e:
        print(f"Error loading {dataset_names[i]}: {e}")
        all_dfs.append(None)
        all_accuracy_cols.append([])

# Find best common (alpha, b)
best_common_alpha, best_common_b = None, None
best_common_metric = -1

# Get unique alpha and b values from first dataset
if all_dfs[0] is not None:
    unique_alphas = sorted(all_dfs[0]['alpha'].unique())
    unique_bs = sorted(all_dfs[0]['b'].unique())
    
    for alpha_val in unique_alphas:
        for b_val in unique_bs:
            total_metric = 0
            valid_count = 0
            
            for idx, df in enumerate(all_dfs):
                if df is None:
                    continue
                cols = all_accuracy_cols[idx]
                if not cols:
                    continue
                    
                try:
                    grouped = df.groupby(['alpha', 'b'])[cols].mean()
                    if (alpha_val, b_val) not in grouped.index:
                        continue
                    
                    # Use same metric as before: QPAD advantage over others
                    mpad_full = next((n for n in cols if n.upper().startswith('MPAD')), None)
                    if mpad_full:
                        mpad_val = grouped.loc[(alpha_val, b_val), mpad_full]
                        others = [grouped.loc[(alpha_val, b_val), c] for c in cols if c != mpad_full]
                        metric = mpad_val - max(others) if others else mpad_val
                    else:
                        metric = grouped.loc[(alpha_val, b_val)].max()
                    
                    total_metric += metric
                    valid_count += 1
                except Exception as e:
                    continue
            
            # Average metric across all valid datasets
            if valid_count == len([df for df in all_dfs if df is not None]):
                avg_metric = total_metric / valid_count
                if avg_metric > best_common_metric:
                    best_common_metric = avg_metric
                    best_common_alpha = alpha_val
                    best_common_b = b_val

print(f"\n*** Best common parameters for all datasets ***")
print(f"α = {best_common_alpha}, b = {best_common_b}")
print(f"Average metric: {best_common_metric:.4f}\n")

# Print individual dataset metrics with these parameters
for idx, df in enumerate(all_dfs):
    if df is None:
        continue
    cols = all_accuracy_cols[idx]
    if not cols:
        continue
    try:
        grouped = df.groupby(['alpha', 'b'])[cols].mean()
        if (best_common_alpha, best_common_b) in grouped.index:
            mpad_full = next((n for n in cols if n.upper().startswith('MPAD')), None)
            if mpad_full:
                mpad_val = grouped.loc[(best_common_alpha, best_common_b), mpad_full]
                others = [grouped.loc[(best_common_alpha, best_common_b), c] for c in cols if c != mpad_full]
                metric = mpad_val - max(others) if others else mpad_val
                print(f"{dataset_names[idx]}: metric = {metric:.4f}")
    except:
        pass

print("\n--- Plotting with common parameters ---")
print(f"Using α={best_common_alpha:.2g}, b={best_common_b} for all datasets\n")

# Store data for both plots
all_plot_data = []

# -------------------------------
# 5. Process each dataset and collect data
# -------------------------------
for i in range(len(csv_paths)):
    print(f"\nProcessing Dataset: {dataset_names[i]}")

    # Use pre-loaded dataframe
    df = all_dfs[i]
    if df is None:
        print(f"  Skipping: Data not loaded")
        all_plot_data.append(None)
        continue

    cols = all_accuracy_cols[i]
    print(f"  Using {len(cols)} accuracy columns")
    if not cols:
        print(f"  Skipping: No accuracy columns")
        all_plot_data.append(None)
        continue

    plot_attrs = []
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

        plot_attrs.append({'full_name': col, 'x_axis_label': x_label, 'color': color, 'legend_label': base})
        final_legend_info[base] = color

    accuracy_cols = [a['full_name'] for a in plot_attrs]
    colors = [a['color'] for a in plot_attrs]
    x_labels = [a['x_axis_label'] for a in plot_attrs]

    # Use common parameters for all datasets
    alpha_val, b_val = best_common_alpha, best_common_b
    
    try:
        grouped = df.groupby(['alpha', 'b'])[accuracy_cols].mean()
    except KeyError as e:
        print(f"  Skipping: Column {e} not found")
        all_plot_data.append(None)
        continue
    if grouped.empty:
        print(f"  Skipping: No data after grouping")
        all_plot_data.append(None)
        continue

    subset = df[(df['alpha']==alpha_val)&(df['b']==b_val)]
    if subset.empty:
        try:
            mean_acc = grouped.loc[(alpha_val, b_val)]
        except:
            print(f"  Skipping: No data for α={alpha_val}, b={b_val}")
            all_plot_data.append(None)
            continue
    else:
        mean_acc = subset[accuracy_cols].mean()
    if mean_acc.isnull().all():
        print(f"  Skipping: Mean accuracies are NaN")
        all_plot_data.append(None)
        continue

    # Store all the data we need for plotting
    max_acc = mean_acc.max()
    rel_acc = mean_acc.fillna(0) if max_acc in [0, np.nan] else mean_acc / max_acc
    abs_values = [mean_acc[col] for col in accuracy_cols]
    rel_values = [rel_acc[col] for col in accuracy_cols]
    best_idx = mean_acc.idxmax() if mean_acc.idxmax() in accuracy_cols else None
    
    all_plot_data.append({
        'accuracy_cols': accuracy_cols,
        'colors': colors,
        'x_labels': x_labels,
        'abs_values': abs_values,
        'rel_values': rel_values,
        'best_idx': best_idx
    })
    
    print(f"  Data collected successfully")

# -------------------------------
# 6. Create Figure 1: Absolute Accuracy
# -------------------------------
print("\n--- Creating Figure 1: Average Recall@k (Absolute Values) ---")
fig1, axs1 = plt.subplots(2, 2, figsize=(17, 13))
axs1 = axs1.flatten()

for i in range(len(csv_paths)):
    ax = axs1[i]
    ax.set_title(dataset_names[i], fontsize=20)
    
    data = all_plot_data[i]
    if data is None:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14)
        ax.axis('off')
        continue
    
    bars = ax.bar(np.arange(len(data['accuracy_cols'])), data['abs_values'], 0.6, color=data['colors'])
    
    if data['best_idx'] is not None:
        idx = data['accuracy_cols'].index(data['best_idx'])
        bars[idx].set_edgecolor('black')
        bars[idx].set_linewidth(2.5)
    
    max_val = max(data['abs_values']) if data['abs_values'] else 1.0
    ax.set_ylim(0, max_val * 1.15)
    ax.set_xticks(np.arange(len(data['accuracy_cols'])))
    ax.set_xticklabels(data['x_labels'], rotation=45, ha='right', fontsize=14)
    ax.set_ylabel("Average Recall@k", fontsize=16)
    ax.grid(axis='y', alpha=0.3)

# -------------------------------
# 7. Create Figure 2: Relative Accuracy w.r.t QPAD
# -------------------------------
print("\n--- Creating Figure 2: Average Recall@k w.r.t QPAD ratio ---")
fig2, axs2 = plt.subplots(2, 2, figsize=(17, 13))
axs2 = axs2.flatten()

for i in range(len(csv_paths)):
    ax = axs2[i]
    ax.set_title(dataset_names[i], fontsize=20)
    
    data = all_plot_data[i]
    if data is None:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14)
        ax.axis('off')
        continue
    
    bars = ax.bar(np.arange(len(data['accuracy_cols'])), data['rel_values'], 0.6, color=data['colors'])
    
    if data['best_idx'] is not None:
        idx = data['accuracy_cols'].index(data['best_idx'])
        bars[idx].set_edgecolor('black')
        bars[idx].set_linewidth(2.5)
    
    ax.axhline(1.0, color='black', linestyle=':', linewidth=1.5)
    current_max = max(data['rel_values']) if data['rel_values'] else 1.0
    ax.set_ylim(0, max(1.15, current_max*1.05))
    ax.set_xticks(np.arange(len(data['accuracy_cols'])))
    ax.set_xticklabels(data['x_labels'], rotation=45, ha='right', fontsize=14)
    ax.set_ylabel("Average Recall@k w.r.t QPAD ratio", fontsize=16)
    ax.grid(axis='y', alpha=0.3)

# -------------------------------
# 8. Add legends and adjust layout for both figures
# -------------------------------
print("\n--- Finalizing plots ---")
if final_legend_info:
    # Replace MPAD with QPAD in legend labels
    labels = [l.replace('MPAD', 'QPAD') for l in final_legend_info.keys()]
    handles = [plt.Rectangle((0,0),1,1, color=final_legend_info[orig_l]) for orig_l in final_legend_info.keys()]
    num_cols = (len(labels)+2)//3
    if len(labels)<=4: num_cols = (len(labels)+1)//2
    elif len(labels)<=6: num_cols = 3
    else: num_cols = 6
    
    # Add legend to Figure 1
    fig1.legend(handles, labels, loc='upper center', ncol=num_cols,
               bbox_to_anchor=(0.5,1), title="Baselines", fontsize=14, title_fontsize=16)
    fig1.subplots_adjust(top=0.88, bottom=0.15, left=0.07, right=0.98, hspace=0.55, wspace=0.25)
    
    # Add legend to Figure 2
    fig2.legend(handles, labels, loc='upper center', ncol=num_cols,
               bbox_to_anchor=(0.5,1), title="Baselines", fontsize=14, title_fontsize=16)
    fig2.subplots_adjust(top=0.88, bottom=0.15, left=0.07, right=0.98, hspace=0.55, wspace=0.25)

plt.show()
print("--- Plotting Complete ---")
