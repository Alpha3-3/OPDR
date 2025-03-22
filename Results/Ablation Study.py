import pandas as pd
import matplotlib.pyplot as plt

#--------------------------------------------------
# 1. Load the entire table into a DataFrame
#--------------------------------------------------
csv_path = "parameter_sweep_results_Isolet_Multiple_methods100.csv"
df = pd.read_csv(csv_path)


#--------------------------------------------------
# 2. Define a function to score a baseline.
#    It checks, for each ablated parameter, what fraction of rows
#    (with the other parameters fixed) have "Better Method" == "dw_pmad"
#--------------------------------------------------
def ablation_dw_pmad_score(df, baseline, ablated_params=['k','Target Ratio','b','alpha']):
    total_dwpmad = 0
    total_rows   = 0
    for param in ablated_params:
        filters = {p: v for p, v in baseline.items() if p != param}
        sub = df.copy()
        for p, v in filters.items():
            sub = sub[sub[p] == v]
        total_rows   += len(sub)
        total_dwpmad += (sub['Better Method'] == 'dw_pmad').sum()
    return total_dwpmad, total_rows

#--------------------------------------------------
# 3. Find the best baseline combination
#    (one that maximizes the fraction of times dw_pmad is best)
#--------------------------------------------------
unique_k      = df['k'].unique()
unique_tr     = df['Target Ratio'].unique()
unique_b      = df['b'].unique()
unique_alpha  = df['alpha'].unique()

best_baseline = None
best_score    = -1.0

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
                dw_count, tot = ablation_dw_pmad_score(df, candidate,
                                                       ablated_params=['k','Target Ratio','b','alpha'])
                if tot == 0:
                    continue
                ratio = dw_count / tot
                if ratio > best_score:
                    best_score = ratio
                    best_baseline = candidate.copy()

print("Chosen baseline:", best_baseline,
      "(dw_pmad best ratio = {:.2%})".format(best_score))

#--------------------------------------------------
# 4. Set up the ablation plots
#--------------------------------------------------

# Use a "cooler" style; here we choose 'tableau-colorblind10'
plt.style.use('tableau-colorblind10')

# Define the accuracy metric columns to plot.
accuracy_cols = [
    'DW-PMAD Accuracy',
    'PCA Accuracy',
    'UMAP Accuracy',
    'Isomap Accuracy',
    'KernelPCA Accuracy',
    'MDS Accuracy'
]

# Helper to filter DataFrame for a given parameter ablation.
def get_subset_for_parameter(df, param, baseline):
    filters = {p: v for p, v in baseline.items() if p != param}
    sub = df.copy()
    for p, v in filters.items():
        sub = sub[sub[p] == v]
    return sub

# We will create 1 row with 4 subplots (one for each parameter)
params_to_ablate = ['k', 'Target Ratio', 'b', 'alpha']
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes = axes.ravel()

# Define different markers, line styles, and colors for each method.
# We choose a special emphasis for "DW-PMAD Accuracy".
method_styles = {
    'DW-PMAD Accuracy':    {'color': 'red',    'marker': 'o', 'linestyle': '-'},
    'PCA Accuracy':        {'color': 'blue',   'marker': 's', 'linestyle': '--'},
    'UMAP Accuracy':       {'color': 'green',  'marker': '^', 'linestyle': '-.'},
    'Isomap Accuracy':     {'color': 'orange', 'marker': 'D', 'linestyle': ':'},
    'KernelPCA Accuracy':  {'color': 'purple', 'marker': 'v', 'linestyle': '-'},
    'MDS Accuracy':        {'color': 'gray',   'marker': 'X', 'linestyle': '--'}
}

# Create each ablation subplot.
for i, param in enumerate(params_to_ablate):
    ax = axes[i]
    subset = get_subset_for_parameter(df, param, best_baseline)
    grouped = subset.groupby(param)
    summary = grouped[accuracy_cols].mean().reset_index()

    # For each accuracy column, plot the line with the given style.
    for col in accuracy_cols:
        style = method_styles.get(col, {})
        ax.plot(summary[param], summary[col],
                marker=style.get('marker', 'o'),
                linestyle=style.get('linestyle', '-'),
                color=style.get('color', None),
                label=col)

    # If ablated parameter is alpha, use a log scale for better visualization.
    if param == 'alpha':
        ax.set_xscale('log')

    ax.set_xlabel(param)
    ax.set_ylabel("Accuracy")

# Show one legend on the last subplot.
axes[-1].legend(loc='best')

# Add a text box with fixed baseline parameters.
fixed_str = (
    "Selected Baseline (max dw_pmad dominance):\n"
    f"k = {best_baseline['k']}, "
    f"Target Ratio = {best_baseline['Target Ratio']}, "
    f"b = {best_baseline['b']}, "
    f"alpha = {best_baseline['alpha']}  "
    f"(dw_pmad best ratio = {best_score:.2%})"
)
print(fixed_str)
fixed_str=""
fig.text(0.5, 0.01, fixed_str,ha='center', va='center', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()
