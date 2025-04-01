import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#---------------------------
# Control variables
#---------------------------
FIG_WIDTH = 16          # Overall figure width (in inches)
FIG_HEIGHT = 8        # Overall figure height (in inches)
MAIN_WIDTH_RATIOS = [1, 1]  # Relative widths for the 2 columns of subfigures
MAIN_HEIGHT_RATIOS = [1, 1] # Relative heights for the 2 rows of subfigures
SUBPLOT_WIDTH_RATIOS = [1, 1, 1, 1]  # Relative widths for the 4 subplots in each subfigure

#---------------------------
# File paths and dataset names
#---------------------------
csv_paths = [
    "parameter_sweep_results_Fasttext_Multiple_methods.csv",
    "parameter_sweep_results_Isolet_Multiple_methods.csv",
    "parameter_sweep_results_Arcene_Multiple_methods.csv",
    "parameter_sweep_results_PBMC3k_Multiple_methods.csv"
]
dataset_names = ["Fasttext", "Isolet", "MNIST", "PBMC3k"]

#---------------------------
# Accuracy metric columns
#---------------------------
accuracy_cols = [
    'MPAD Accuracy',
    'UMAP Accuracy',
    'Isomap Accuracy',
    'KernelPCA Accuracy',
    'MDS Accuracy'
]

#--------------------------------------------------
# Helper functions (same as your original code)
#--------------------------------------------------
def ablation_dw_pmad_score(df, baseline, ablated_params=['k', 'Target Ratio', 'b', 'alpha']):
    total_dwpmad = 0
    total_rows = 0
    for param in ablated_params:
        filters = {p: v for p, v in baseline.items() if p != param}
        sub = df.copy()
        for p, v in filters.items():
            sub = sub[sub[p] == v]
        total_rows += len(sub)
        total_dwpmad += (sub['Better Method'] == 'dw_pmad').sum()
    return total_dwpmad, total_rows

def ablation_dw_pmad_accuracy(df, baseline, ablated_params=['k', 'Target Ratio', 'b', 'alpha']):
    total_accuracy = 0.0
    count = 0
    for param in ablated_params:
        filters = {p: v for p, v in baseline.items() if p != param}
        sub = df.copy()
        for p, v in filters.items():
            sub = sub[sub[p] == v]
        if len(sub) > 0:
            total_accuracy += sub["MPAD Accuracy"].mean()
            count += 1
    return total_accuracy, count

def meets_target_ratio_condition(df, baseline):
    metrics = [
        'MPAD Accuracy',
        'UMAP Accuracy',
        'Isomap Accuracy',
        'KernelPCA Accuracy',
        'MDS Accuracy'
    ]
    filters = {p: v for p, v in baseline.items() if p != 'Target Ratio'}
    sub = df.copy()
    for p, v in filters.items():
        sub = sub[sub[p] == v]
    groups = sub.groupby('Target Ratio')
    count = 0
    for target_val, group in groups:
        group_mean = group[metrics].mean()
        ranking = group_mean.sort_values(ascending=False)
        rank = list(ranking.index).index('MPAD Accuracy') + 1
        if rank <= 2:
            count += 1
    return count >= 2

def get_subset_for_parameter(df, param, baseline):
    filters = {p: v for p, v in baseline.items() if p != param}
    sub = df.copy()
    for p, v in filters.items():
        sub = sub[sub[p] == v]
    return sub

#---------------------------
# Method styles
#---------------------------
method_styles = {
    'MPAD Accuracy':    {'color': 'red',    'marker': 'o', 'linestyle': '-'},
    'UMAP Accuracy':       {'color': 'green',  'marker': '^', 'linestyle': '-.'},
    'Isomap Accuracy':     {'color': 'orange', 'marker': 'D', 'linestyle': ':'},
    'KernelPCA Accuracy':  {'color': 'purple', 'marker': 'v', 'linestyle': '-'},
    'MDS Accuracy':        {'color': 'gray',   'marker': 'X', 'linestyle': '--'}
}

plt.style.use('tableau-colorblind10')

#--------------------------------------------------
# Create the main figure and define subfigures using a gridspec.
#--------------------------------------------------
main_fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
# Create a gridspec for subfigures with manual width and height ratios.
gs = main_fig.add_gridspec(2, 2, width_ratios=MAIN_WIDTH_RATIOS, height_ratios=MAIN_HEIGHT_RATIOS)
# Create subfigures manually.
subfigs = [[main_fig.add_subfigure(gs[i, j]) for j in range(2)] for i in range(2)]

# Loop over the CSV files/datasets.
for i, csv_path in enumerate(csv_paths):
    df = pd.read_csv(csv_path)
    df = df[~df['b'].isin([40, 50])]
    # Find the best baseline candidate in this dataset.
    unique_k     = [k for k in df['k'].unique() if k != 1]
    unique_tr    = df['Target Ratio'].unique()
    unique_b     = df['b'].unique()
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

    # Select the appropriate subfigure from the grid.
    subfig = subfigs[i // 2][i % 2]
    # Each subfigure will have 1 row and 4 columns (one for each parameter ablation).
    axes = subfig.subplots(1, 4, sharey=True, gridspec_kw={'width_ratios': SUBPLOT_WIDTH_RATIOS})

    params_to_ablate = ['k', 'Target Ratio', 'b', 'alpha']
    for j, param in enumerate(params_to_ablate):
        ax = axes[j]
        subset = get_subset_for_parameter(df, param, best_baseline)
        grouped = subset.groupby(param)
        summary = grouped[accuracy_cols].mean().reset_index()

        for col in accuracy_cols:
            style = method_styles.get(col, {})
            ax.plot(summary[param], summary[col],
                    marker=style.get('marker', 'o'),
                    linestyle=style.get('linestyle', '-'),
                    color=style.get('color', None),
                    label=col)
        if param == 'alpha':
            ax.set_xscale('log')
        ax.set_xlabel(param)
        if j == 0:
            ax.set_ylabel("Accuracy")

    # Add a title to the subfigure with dataset name and baseline parameters.
    baseline_text = (
        f"Baseline: k = {best_baseline['k']}, "
        f"Target Ratio = {best_baseline['Target Ratio']}, "
        f"b = {best_baseline['b']}, "
        f"alpha = {best_baseline['alpha']} (MPAD Acc = {best_score:.2%})"
    )
    subfig.suptitle(f"Dataset: {dataset_names[i]}\n{baseline_text}", fontsize=10)

# After creating your subfigures and plotting everything...
# Create a common legend as before.
handles = []
labels = []
for col in accuracy_cols:
    style = method_styles.get(col, {})
    line, = plt.plot([], [],
                     marker=style.get('marker', 'o'),
                     linestyle=style.get('linestyle', '-'),
                     color=style.get('color', None))
    handles.append(line)
    labels.append(col)

# Instead of using main_fig.legend, add a dedicated axes for the legend.
# The list [left, bottom, width, height] are in figure-relative coordinates.
# Adjust these numbers until the legend appears where you want it.
legend_ax = main_fig.add_axes([0.0, 0.92, 1.0, 0.06])
legend_ax.axis('off')  # Hide the axis lines and ticks.
legend_ax.legend(handles, labels, loc='center', ncol=len(accuracy_cols), fontsize=10)


plt.show()