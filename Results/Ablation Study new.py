import pandas as pd
import matplotlib.pyplot as plt
import math  # for math.ceil
import os  # to handle file paths

# Increase global font size and use colorblind-friendly style
ggs = {'font.size': 14}
plt.rcParams.update(ggs)
plt.style.use('tableau-colorblind10')

#--------------------------------------------------
# 1. List of CSV files and dataset names
#--------------------------------------------------
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

csv_paths = [
    os.path.join(script_dir, "parameter_sweep_results_Fasttext_Multiple_methods_with_additional_baselines.csv"),
    os.path.join(script_dir, "parameter_sweep_results_Isolet_Multiple_methods_with_additional_baselines.csv"),
    os.path.join(script_dir, "parameter_sweep_results_Arcene_Multiple_methods_with_additional_baselines.csv"),
    os.path.join(script_dir, "parameter_sweep_results_PBMC3k_Multiple_methods_with_additional_baselines.csv")
]
dataset_names = ["Fasttext", "Isolet", "Arcene", "PBMC3k"]

#--------------------------------------------------
# 2. Display config and predefined colors/names
#--------------------------------------------------
display_methods_config = [
    {'x_label': 'QPAD',    'full_csv_col_name': 'MPAD Accuracy'},
    {'x_label': 'UMAP',    'full_csv_col_name': 'UMAP Accuracy'},
    {'x_label': 'Isomap',  'full_csv_col_name': 'Isomap Accuracy'},
    {'x_label': 'KPCA',    'full_csv_col_name': 'KernelPCA Accuracy'},
    {'x_label': 'AE',      'full_csv_col_name': 'Autoencoder Accuracy'},
    {'x_label': 'FeatAgg', 'full_csv_col_name': 'Feature Agglomeration Accuracy'},
    {'x_label': 'LLE',     'full_csv_col_name': 'LLE Accuracy'},
    {'x_label': 'NMF',     'full_csv_col_name': 'NMF Accuracy'},
    {'x_label': 'RandProj','full_csv_col_name': 'Random Projection Accuracy'},
    {'x_label': 'VAE',     'full_csv_col_name': 'VAE Accuracy'},
    {'x_label': 'tSNE',    'full_csv_col_name': 't-SNE Accuracy'},
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

# Manual marker and linestyle for each method to ensure uniqueness
manual_marker_linestyles = {
    'MPAD Accuracy':          {'marker': 'o',  'linestyle': '-'},
    'UMAP Accuracy':          {'marker': '^',  'linestyle': '-.'},
    'Isomap Accuracy':        {'marker': 'D',  'linestyle': ':'},
    'KernelPCA Accuracy':     {'marker': 'v',  'linestyle': '-'},
    'Autoencoder Accuracy':   {'marker': 'x',  'linestyle': '--'},
    'Feature Agglomeration Accuracy': {'marker': 's',  'linestyle': '-.'},
    'LLE Accuracy':           {'marker': 'p',  'linestyle': ':'},
    'NMF Accuracy':           {'marker': 'h',  'linestyle': '-'},
    'Random Projection Accuracy': {'marker': '*',  'linestyle': '--'},
    'VAE Accuracy':           {'marker': '+',  'linestyle': '-.'},
    't-SNE Accuracy':         {'marker': 'X',  'linestyle': ':'},
    'LSH Accuracy':           {'marker': '<',  'linestyle': '--'}
}

def normalize_col(name: str) -> str:
    """Remove spaces and hyphens to match keys robustly."""
    return name.replace(' ', '').replace('-', '')

# Build method_styles by normalizing keys
method_styles = {}
default_style = {'color': None, 'marker': 'o', 'linestyle': '-'}
for conf in display_methods_config:
    full = conf['full_csv_col_name']
    key_norm = normalize_col(full)
    # start with manual marker/linestyle or default
    style = manual_marker_linestyles.get(full, default_style.copy()).copy()
    # assign predefined color
    style['color'] = predefined_method_attributes.get(full, {}).get('color', None)
    # store short_name
    style['short_name'] = conf['x_label']
    method_styles[key_norm] = style

#--------------------------------------------------
# 3. Ablation helper functions (unchanged)
#--------------------------------------------------
def ablation_dw_pmad_accuracy(df, baseline, ablated_params=['k', 'Target Ratio', 'b', 'alpha']):
    total, cnt = 0.0, 0
    for p in ablated_params:
        filters = {q: v for q, v in baseline.items() if q != p}
        sub = df.copy()
        for q, v in filters.items(): sub = sub[sub[q] == v]
        if len(sub):
            total += sub['MPAD Accuracy'].mean()
            cnt += 1
    return total, cnt


def find_best_baseline(df):
    def uniq(col): return [x for x in df[col].unique() if not (col=='k' and x==1)]
    best, best_score = None, -1
    for k in uniq('k'):
        for tr in uniq('Target Ratio'):
            for b in uniq('b'):
                for a in uniq('alpha'):
                    cand = {'k':k,'Target Ratio':tr,'b':b,'alpha':a}
                    tot,cnt = ablation_dw_pmad_accuracy(df, cand)
                    if cnt and (score:=tot/cnt)>=0.5 and score>best_score:
                        best_score,best = score,cand.copy()
    return best, best_score


def get_subset_for_parameter(df, param, baseline):
    filters = {q: v for q, v in baseline.items() if q != param}
    sub = df.copy()
    for q, v in filters.items(): sub = sub[sub[q] == v]
    return sub

#--------------------------------------------------
# 4. Find common best baseline across all datasets
#--------------------------------------------------
print("=== Finding common best baseline across all datasets ===")
# Load all datasets first
all_dfs = []
for idx, path in enumerate(csv_paths):
    df = pd.read_csv(path)
    df = df[~df['b'].isin([40, 50])]
    all_dfs.append(df)
    print(f"Loaded {dataset_names[idx]}: {len(df)} rows")

# Get unique parameter values from first dataset (assuming all have same parameters)
df_ref = all_dfs[0]
def uniq(col): return [x for x in df_ref[col].unique() if not (col=='k' and x==1)]
unique_k = uniq('k')
unique_tr = uniq('Target Ratio')
unique_b = uniq('b')
unique_alpha = uniq('alpha')

# Search for best common baseline
best_common_baseline = None
best_common_score = -1

for k in unique_k:
    for tr in unique_tr:
        for b in unique_b:
            for a in unique_alpha:
                cand = {'k': k, 'Target Ratio': tr, 'b': b, 'alpha': a}
                
                # Calculate average score across all datasets
                total_score = 0.0
                valid_datasets = 0
                
                for idx, df in enumerate(all_dfs):
                    tot, cnt = ablation_dw_pmad_accuracy(df, cand)
                    if cnt > 0:
                        score = tot / cnt
                        if score >= 0.5:  # Only consider if score is reasonable
                            total_score += score
                            valid_datasets += 1
                
                # Average score across all valid datasets
                if valid_datasets == len(all_dfs):  # All datasets must be valid
                    avg_score = total_score / valid_datasets
                    if avg_score > best_common_score:
                        best_common_score = avg_score
                        best_common_baseline = cand.copy()

print(f"\n*** Best common baseline for all datasets ***")
print(f"Parameters: k={best_common_baseline['k']}, DRR={best_common_baseline['Target Ratio']:.1f}, "
      f"b={best_common_baseline['b']}, α={best_common_baseline['alpha']:.0f}")
print(f"Average score: {best_common_score:.2%}\n")

# Print individual dataset scores with this baseline
for idx, df in enumerate(all_dfs):
    tot, cnt = ablation_dw_pmad_accuracy(df, best_common_baseline)
    if cnt > 0:
        print(f"{dataset_names[idx]}: {tot/cnt:.2%}")

#--------------------------------------------------
# 5. Plotting setup and execution
#--------------------------------------------------
n_datasets = len(csv_paths)
params_to_ablate = ['k', 'Target Ratio', 'b', 'alpha']
param_display_names = {'k': 'k', 'Target Ratio': 'DRR', 'b': 'b', 'alpha': 'α'}
# Set width ratios: alpha column gets more space (1.5x) to avoid label overlap
fig, axes = plt.subplots(n_datasets, len(params_to_ablate), figsize=(24, 16),
                         gridspec_kw={'width_ratios': [1, 1, 1, 1.5]})
all_handles = {}

for idx, df in enumerate(all_dfs):
    # Normalize column keys for lookup
    acc_cols = [c for c in df.columns if c.endswith('Accuracy') and c not in ['PCA Accuracy','FastICA Accuracy','MDS Accuracy']]
    
    # Use the common baseline for all datasets
    best_base = best_common_baseline
    tot, cnt = ablation_dw_pmad_accuracy(df, best_base)
    best_score = tot / cnt if cnt > 0 else 0
    print(f"\n{dataset_names[idx]} – using common baseline, score {best_score:.2%}")

    for j, param in enumerate(params_to_ablate):
        ax = axes[idx, j]
        summary = get_subset_for_parameter(df, param, best_base).groupby(param)[acc_cols].mean().reset_index()
        for col in acc_cols:
            key_norm = normalize_col(col)
            style = method_styles.get(key_norm, default_style)
            label = style.get('short_name', col.replace(' Accuracy',''))
            line, = ax.plot(
                summary[param], summary[col],
                marker=style['marker'], linestyle=style['linestyle'], color=style['color'], label=label
            )
            all_handles[label] = line

        if param == 'alpha':
            ax.set_xscale('log')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        else:
            plt.setp(ax.get_xticklabels(), rotation=0, fontsize=12)

        # Use display name for x-axis label
        display_name = param_display_names.get(param, param)
        ax.set_xlabel(display_name, fontsize=16)
        ax.set_ylabel('Recall@k', fontsize=16)
        ax.set_xticks(summary[param])
        ax.set_xticklabels([f'{x:.2f}' if isinstance(x, float) else x for x in summary[param]], fontsize=12)

        if j == 0:
            ax.set_title(
                f"{dataset_names[idx]} | baseline k={best_base['k']}, DRR={best_base['Target Ratio']:.1f}, "
                f"b={best_base['b']}, α={best_base['alpha']:.0f}",
                loc='left', fontsize=16
            )

#--------------------------------------------------
# 6. Global legend & layout adjustments
#--------------------------------------------------
num_items = len(all_handles)
ncol = math.ceil(num_items / 2) if num_items else 1
fig.legend(
    all_handles.values(), all_handles.keys(),
    loc='upper center', ncol=ncol, fontsize=16
)
plt.subplots_adjust(
    top=0.92,
    bottom=0.05,
    left=0.03,
    right=0.998,
    hspace=0.39,
    wspace=0.16
)
plt.show()
