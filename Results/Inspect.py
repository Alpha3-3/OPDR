import pandas as pd

# Load the data
df = pd.read_csv("parameter_sweep_results_CIFAR-10.csv")

# Aggregate performance across different target_ratio and k for each (alpha, b)
global_results = df.groupby(['alpha', 'b'])['DW-PMAD Accuracy'].mean().reset_index()

# Sort to get the best-performing combination based on DW-PMAD Accuracy
global_optimal = global_results.sort_values(by='DW-PMAD Accuracy', ascending=False).iloc[0]

# Extract the optimal parameter values
alpha_opt = global_optimal['alpha']
b_opt = global_optimal['b']
print("Global optimal parameter combination (alpha, b):", alpha_opt, b_opt)

# Filter the original DataFrame to show only rows with these optimal parameters
optimal_rows = df[(df['alpha'] == alpha_opt) & (df['b'] == b_opt)]

# Define the columns with the different method accuracies
accuracy_columns = [
    "DW-PMAD Accuracy",
    "PCA Accuracy",
    #"UMAP Accuracy",
    #"Isomap Accuracy",
    #"KernelPCA Accuracy",
    #"MDS Accuracy"
]

# For each row, determine which method has the highest accuracy.
# This adds a new column 'Best Method' with the column name that achieved the maximum.
optimal_rows["Best Method"] = optimal_rows[accuracy_columns].idxmax(axis=1)

# Optionally, if you also want to know the best accuracy value:
optimal_rows["Best Accuracy"] = optimal_rows[accuracy_columns].max(axis=1)

print("Relevant rows with the optimal parameter combination and best method:")
print(optimal_rows.to_string())

### Plotting
import matplotlib.pyplot as plt
import numpy as np

# Sort the optimal rows based on Target Ratio first, then by k
optimal_rows_sorted = optimal_rows.sort_values(by=["Target Ratio", "k"])

# Define the accuracy columns
accuracy_columns = [
    "DW-PMAD Accuracy",
    "PCA Accuracy",
    #"UMAP Accuracy",
    #"Isomap Accuracy",
    #"KernelPCA Accuracy",
    #"MDS Accuracy"
]

num_rows = optimal_rows_sorted.shape[0]
num_methods = len(accuracy_columns)
x = np.arange(num_rows)  # one group per row
width = 0.13  # width for each bar

fig, ax = plt.subplots(figsize=(12, 6))

# Plot a bar for each method in each group
for i, col in enumerate(accuracy_columns):
    accuracies = optimal_rows_sorted[col]
    # Calculate the horizontal offset for each method's bar within each group.
    offset = x - width * num_methods/2 + i * width + width/2
    bars = ax.bar(offset, accuracies, width, label=col)

    # For each bar, check if it corresponds to the best method.
    for j, bar in enumerate(bars):
        if optimal_rows_sorted.iloc[j]["Best Method"] == col:
            # Highlight the bar with a black border.
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
            # If the best method is DW-PMAD, add a special red star on top.
            if col == "DW-PMAD Accuracy":
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,  # vertical offset; adjust as needed
                    '*',
                    ha='center', va='bottom',
                    color='red', fontsize=14
                )

# Label the x-axis using the sorted values: Target Ratio and k.
xtick_labels = [f"TR={row['Target Ratio']}\nk={row['k']}" for _, row in optimal_rows_sorted.iterrows()]
ax.set_xticks(x)
ax.set_xticklabels(xtick_labels)

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison for Optimal Parameter Combination Rows\n(Special Mark for DW-PMAD as Best Method)')
ax.legend()

plt.tight_layout()
plt.show()
