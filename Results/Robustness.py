import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------
# 1. Load the CSV table
# ---------------------------------------------------
csv_path = "parameter_sweep_results_Fasttext_Multiple_methods1.csv"
df = pd.read_csv(csv_path)
# We assume Dimension==298 throughout.
df = df[df['Dimension'] == 298].copy()

# ---------------------------------------------------
# 2. How is the average accuracy calculated?
# ---------------------------------------------------
# For any given group (for example, when we fix a baseline combination)
# the "average accuracy" for a method is simply the arithmetic mean of that
# method's accuracy values across all the rows in that group.
#
# For example, if we group by a parameter (say k) and then compute:
#   group_mean = group['DW-PMAD Accuracy'].mean()
# that mean is computed as:
#
#      (value1 + value2 + ... + valueN) / N
#
# where N is the number of rows in that group.

# ---------------------------------------------------
# 3. Count "Best" and "Second Best" Occurrences Over the Entire Parameter Space
# ---------------------------------------------------
# We assume the following columns hold the accuracy values for each method.
accuracy_cols = [
    'DW-PMAD Accuracy',
    'PCA Accuracy',
    'UMAP Accuracy',
    'Isomap Accuracy',
    'KernelPCA Accuracy',
    'MDS Accuracy'
]

# We will now loop over every row to determine which method is best and second best.
# (We use the built-in pandas ranking method on each row's accuracy values,
#  ranking in descending order so that rank 1 means highest accuracy.)

methods = accuracy_cols
best_counts = {m: 0 for m in methods}
second_counts = {m: 0 for m in methods}

for idx, row in df.iterrows():
    accuracies = row[accuracy_cols]
    # Rank in descending order: highest value gets rank 1.
    ranks = accuracies.rank(method='min', ascending=False)
    # The best method is the one with the minimum rank.
    best_method = ranks.idxmin()
    # Now, determine the second best: choose the method with rank==2.
    # (If there is a tie, this simply selects one arbitrarily.)
    second_method = None
    for m in methods:
        if ranks[m] == 2:
            second_method = m
            break
    best_counts[best_method] += 1
    if second_method is not None:
        second_counts[second_method] += 1

# Create a DataFrame of the counts for easier plotting.
counts_df = pd.DataFrame({
    'Method': methods,
    'Best Count': [best_counts[m] for m in methods],
    'Second Best Count': [second_counts[m] for m in methods]
})

# ---------------------------------------------------
# 4. Plot the "Robustness" Graph
#    This graph will show, for each method, the number of times it achieved the best and second best accuracy.
# ---------------------------------------------------

# We define custom styles to emphasize DW-PMAD:
method_styles = {
    'DW-PMAD Accuracy':    {'color': 'red'},       # Emphasize in red
    'PCA Accuracy':        {'color': 'blue'},
    'UMAP Accuracy':       {'color': 'green'},
    'Isomap Accuracy':     {'color': 'orange'},
    'KernelPCA Accuracy':  {'color': 'purple'},
    'MDS Accuracy':        {'color': 'gray'}
}

# Create a grouped bar chart.
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(methods))
width = 0.35

# Bar for "Best" counts
bars1 = ax.bar(x - width/2, counts_df['Best Count'], width,
               label='Best', color=[method_styles[m]['color'] for m in methods])

# Bar for "Second Best" counts
bars2 = ax.bar(x + width/2, counts_df['Second Best Count'], width,
               label='Second Best', color='lightgray')

# Optionally, we can add a black border to the DW-PMAD best bar to further emphasize.
dw_idx = methods.index('DW-PMAD Accuracy')
bars1[dw_idx].set_edgecolor('black')
bars1[dw_idx].set_linewidth(2.5)

# Labeling the plot.
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Robustness of Methods: Frequency as Best and Second Best", fontsize=14)
ax.legend(loc='best')

plt.tight_layout()
plt.show()
