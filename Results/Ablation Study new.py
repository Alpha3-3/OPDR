import pandas as pd
import matplotlib.pyplot as plt

# Increase global font size
plt.rcParams.update({'font.size': 14})

#--------------------------------------------------
# 1. List of CSV files and manual dataset names
#--------------------------------------------------
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

#--------------------------------------------------
# 2. Define functions for scoring and baseline selection
#--------------------------------------------------
def ablation_dw_pmad_accuracy(df, baseline, ablated_params=['k', 'Target Ratio', 'b', 'alpha']):
    total_accuracy = 0.0
    count = 0
    for param in ablated_params:
        # Filter using all parameters except the one being ablated.
        filters = {p: v for p, v in baseline.items() if p != param}
        sub = df.copy()
        for p, v in filters.items():
            sub = sub[sub[p] == v]
        if len(sub) > 0:
            total_accuracy += sub["MPAD Accuracy"].mean()
            count += 1
    return total_accuracy, count

def find_best_baseline(df):
    unique_k = [k for k in df['k'].unique() if k != 1]
    unique_tr = df['Target Ratio'].unique()
    unique_b = df['b'].unique()
    unique_alpha = df['alpha'].unique()
    best_baseline = None
    best_score = -1.0
    for k_ in unique_k:
        for tr_ in unique_tr:
            for b_ in unique_b:
                for alpha_ in unique_alpha:
                    candidate = {
                        'k': k_,
                        'Target Ratio': tr_,
                        'b': b_,
                        'alpha': alpha_
                    }
                    total_accuracy, cnt = ablation_dw_pmad_accuracy(df, candidate,
                                                                    ablated_params=['k', 'Target Ratio', 'b', 'alpha'])
                    if cnt == 0:
                        continue
                    candidate_accuracy = total_accuracy / cnt
                    if candidate_accuracy < 0.5:
                        continue
                    if candidate_accuracy > best_score:
                        best_score = candidate_accuracy
                        best_baseline = candidate.copy()
    return best_baseline, best_score

# Helper to filter DataFrame for a given parameter ablation.
def get_subset_for_parameter(df, param, baseline):
    filters = {p: v for p, v in baseline.items() if p != param}
    sub = df.copy()
    for p, v in filters.items():
        sub = sub[sub[p] == v]
    return sub

#--------------------------------------------------
# 3. Define accuracy metrics and styles (ignoring PCA Accuracy)
#--------------------------------------------------
accuracy_cols = [
    'MPAD Accuracy',
    'UMAP Accuracy',
    'Isomap Accuracy',
    'KernelPCA Accuracy',
    'MDS Accuracy'
]

method_styles = {
    'MPAD Accuracy':    {'color': 'red',    'marker': 'o', 'linestyle': '-'},
    'UMAP Accuracy':    {'color': 'green',  'marker': '^', 'linestyle': '-.'},
    'Isomap Accuracy':  {'color': 'orange', 'marker': 'D', 'linestyle': ':'},
    'KernelPCA Accuracy': {'color': 'purple', 'marker': 'v', 'linestyle': '-'},
    'MDS Accuracy':     {'color': 'gray',   'marker': 'X', 'linestyle': '--'}
}

#--------------------------------------------------
# 4. Set up the overall figure:
#    2 rows and 8 columns (each dataset occupies 4 contiguous subplots).
#    Adjust figure size and spacing to reduce empty spaces.
#--------------------------------------------------
plt.style.use('tableau-colorblind10')
n_datasets = len(csv_paths)
datasets_per_row = 2  # two datasets per row
params_to_ablate = ['k', 'Target Ratio', 'b', 'alpha']
n_rows = 2
n_cols = datasets_per_row * len(params_to_ablate)  # 2 * 4 = 8 columns

fig_width = 22
fig_height = 8
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

# For a global legend.
all_handles = {}

#--------------------------------------------------
# 5. Process each dataset and create its ablation subplots
#--------------------------------------------------
for idx, csv_path in enumerate(csv_paths):
    # Determine the grid position for this dataset.
    grid_row = idx // datasets_per_row
    col_offset = (idx % datasets_per_row) * len(params_to_ablate)

    # Load dataset and ignore rows with b = 40 or 50.
    df = pd.read_csv(csv_path)
    df = df[~df['b'].isin([40, 50])]

    # Determine the best baseline for this dataset.
    best_baseline, best_score = find_best_baseline(df)
    if best_baseline is None:
        print(f"No baseline candidate found for {csv_path}. Skipping dataset.")
        continue
    print(f"{dataset_names[idx]} -- Chosen baseline: {best_baseline} (MPAD Accuracy = {best_score:.2%})")

    # Loop over each ablated parameter (one subplot per parameter).
    for i, param in enumerate(params_to_ablate):
        ax = axes[grid_row, col_offset + i]
        subset = get_subset_for_parameter(df, param, best_baseline)
        grouped = subset.groupby(param)
        summary = grouped[accuracy_cols].mean().reset_index()

        # Plot each accuracy column.
        for acc_col in accuracy_cols:
            style = method_styles.get(acc_col, {})
            line, = ax.plot(
                summary[param],
                summary[acc_col],
                marker=style.get('marker', 'o'),
                linestyle=style.get('linestyle', '-'),
                color=style.get('color', None),
                label=acc_col
            )
            if acc_col not in all_handles:
                all_handles[acc_col] = line

        # Use log scale for alpha.
        if param == 'alpha':
            ax.set_xscale('log')
        ax.set_xlabel(param, fontsize=16)
        ax.set_ylabel("Accuracy", fontsize=16)

        # Only on the left-most subplot for this dataset, add the dataset name and baseline info.
        if i == 0:
            title_text = (f"{dataset_names[idx]}  Baseline:  "
                          f"k = {best_baseline['k']}, TR = {best_baseline['Target Ratio']}, "
                          f"b = {best_baseline['b']}, alpha = {best_baseline['alpha']}  "
                          f"(MPAD: {best_score:.2%})")
            ax.set_title(title_text, loc='left', fontsize=16)

#--------------------------------------------------
# 6. Create one global legend and adjust layout to reduce empty spaces
#--------------------------------------------------
fig.legend(all_handles.values(), all_handles.keys(), loc='upper right', ncol=len(accuracy_cols), fontsize=16)
plt.subplots_adjust(top=0.895,
                    bottom=0.075,
                    left=0.035,
                    right=0.995,
                    hspace=0.33,
                    wspace=0.39)
plt.show()
