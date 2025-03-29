import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Load the data.
# -------------------------------
csv_path = "parameter_sweep_results_Isolet_Multiple_methods.csv"
df = pd.read_csv(csv_path)

# -------------------------------
# 2. Define the accuracy columns (methods).
# -------------------------------
accuracy_cols = [
    'DW-PMAD Accuracy',
    'PCA Accuracy',
    'UMAP Accuracy',
    'Isomap Accuracy',
    'KernelPCA Accuracy',
    'MDS Accuracy'
]

# -------------------------------
# 3. Compute the best (alpha, b) pair based on DW-PMAD superiority.
#    For each (alpha, b) combination, we compute the mean accuracy per method (averaged over other params),
#    then compute the gap = (DW-PMAD Accuracy) - (max of all other methods).
# -------------------------------
grouped = df.groupby(['alpha', 'b'])[accuracy_cols].mean().reset_index()

def compute_gap(row):
    dw = row['DW-PMAD Accuracy']
    others = [row[col] for col in accuracy_cols if col != 'DW-PMAD Accuracy']
    return dw - max(others)

grouped['gap'] = grouped.apply(compute_gap, axis=1)

best_idx = grouped['gap'].idxmax()
best_combo = grouped.loc[best_idx, ['alpha', 'b']]
print("Best (alpha, b) combination:", best_combo.to_dict())
print("Superiority gap = {:.2%}".format(grouped.loc[best_idx, 'gap']))

# -------------------------------
# 4. For the best (alpha, b), get overall average accuracy (across all other parameter values).
# -------------------------------
mask = (df['alpha'] == best_combo['alpha']) & (df['b'] == best_combo['b'])
best_data = df[mask]
mean_acc = best_data[accuracy_cols].mean()

# -------------------------------
# 5. Plot a grouped bar chart to compare the methods.
#    Use a style that is visually compelling (here we use tableau-colorblind10).
#    We set different colors and emphasize DW-PMAD.
# -------------------------------
plt.style.use('tableau-colorblind10')

fig, ax = plt.subplots(figsize=(8, 6))

methods = accuracy_cols
x = np.arange(len(methods))
width = 0.6

# Define a style for each method; emphasis for DW-PMAD:
method_styles = {
    'DW-PMAD Accuracy':    {'color': 'red'},
    'PCA Accuracy':        {'color': 'blue'},
    'UMAP Accuracy':       {'color': 'green'},
    'Isomap Accuracy':     {'color': 'orange'},
    'KernelPCA Accuracy':  {'color': 'purple'},
    'MDS Accuracy':        {'color': 'gray'}
}

colors = [method_styles[m]['color'] for m in methods]

bars = ax.bar(x, mean_acc, width, color=colors)

# Emphasize DW-PMAD by adding a thick black edge.
dw_idx = methods.index('DW-PMAD Accuracy')
bars[dw_idx].set_edgecolor('black')
bars[dw_idx].set_linewidth(2.5)

# Annotate each bar with its value.
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # vertical offset in points
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.set_ylabel("Average Accuracy")
ax.set_title("Performance Comparison for Best (alpha, b) Combination")

# Add a text annotation below the plot showing the fixed baseline values.
baseline_text = (
    f"Selected Best Parameters:\n"
    f"alpha = {best_combo['alpha']}, b = {best_combo['b']}\n"
    f"(DW-PMAD superiority gap = {grouped.loc[best_idx, 'gap']:.2%})"
)
ax.text(0.5, -0.18, baseline_text, transform=ax.transAxes, ha='center', va='center', fontsize=10)

plt.tight_layout()
plt.show()
