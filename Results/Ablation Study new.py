import pandas as pd
import matplotlib.pyplot as plt
import math  # for math.ceil

# Increase global font size and use colorblind-friendly style
ggs = {'font.size': 14}
plt.rcParams.update(ggs)
plt.style.use('tableau-colorblind10')

#--------------------------------------------------
# 1. List of CSV files and dataset names
#--------------------------------------------------
csv_paths = [
    "parameter_sweep_results_Fasttext_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_Isolet_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_Arcene_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_PBMC3k_Multiple_methods_with_additional_baselines.csv"
]
dataset_names = ["Fasttext", "Isolet", "Arcene", "PBMC3k"]

#--------------------------------------------------
# 2. Display config and predefined colors/names
#--------------------------------------------------
display_methods_config = [
    {'x_label': 'MPAD',    'full_csv_col_name': 'MPAD Accuracy'},
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
# 4. Plotting setup and execution
#--------------------------------------------------
n_datasets = len(csv_paths)
params_to_ablate = ['k', 'Target Ratio', 'b', 'alpha']
fig, axes = plt.subplots(n_datasets, len(params_to_ablate), figsize=(24, 16))
all_handles = {}

for idx, path in enumerate(csv_paths):
    df = pd.read_csv(path)
    df = df[~df['b'].isin([40, 50])]
    # Normalize column keys for lookup
    acc_cols = [c for c in df.columns if c.endswith('Accuracy') and c not in ['PCA Accuracy','FastICA Accuracy','MDS Accuracy']]
    best_base, best_score = find_best_baseline(df)
    print(f"{dataset_names[idx]} – baseline {best_base}, score {best_score:.2%}")

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

        ax.set_xlabel(param, fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_xticks(summary[param])
        ax.set_xticklabels([f'{x:.2f}' if isinstance(x, float) else x for x in summary[param]], fontsize=12)

        if j == 0:
            ax.set_title(
                f"{dataset_names[idx]} | baseline k={best_base['k']}, TR={best_base['Target Ratio']:.1f}, "
                f"b={best_base['b']}, α={best_base['alpha']:.0f}",
                loc='left', fontsize=16
            )

#--------------------------------------------------
# 5. Global legend & layout adjustments
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
