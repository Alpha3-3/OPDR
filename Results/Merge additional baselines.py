import pandas as pd

# File names (adjust paths if necessary)
main_csv = 'parameter_sweep_results_Isolet_Multiple_methods_with_additional_baselines.csv'
add_csv = '../More Baseline/lsh_baseline_results_Isolet.csv'
output_csv = 'parameter_sweep_results_Isolet_Multiple_methods_with_additional_baselines.csv'

# Read the main parameter sweep CSV file
df_main = pd.read_csv(main_csv)

# Read the additional baselines CSV file
df_add = pd.read_csv(add_csv)

# Compute the target ratio in the additional file.
# This is calculated as TargetDim / OriginalDim.
# We round it to 2 decimals to ensure that it matches the main file's "Target Ratio".
df_add['Target Ratio'] = (df_add['TargetDim'] / df_add['OriginalDim']).round(2)

# Optional: Also round the main file's "Target Ratio" to avoid floating point issues.
df_main['Target Ratio'] = df_main['Target Ratio'].round(2)

# Pivot the additional baselines data so that each method becomes a pair of columns.
# We want one column for the Accuracy and another for the Method Time, per method.
# The pivot table will use 'Target Ratio' and 'k' as the merging keys.
pivot_df = df_add.pivot_table(
    index=['Target Ratio', 'k'],
    columns='Method',
    values=['Accuracy', 'MethodTime(s)']
)

# Flatten the multi-index columns into single-level column names such as:
# "RandomProjection Accuracy" and "RandomProjection MethodTime(s)".
pivot_df.columns = [f'{method} {metric}' for metric, method in pivot_df.columns]
pivot_df = pivot_df.reset_index()

# Merge the pivoted additional results with the main DataFrame based on "Target Ratio" and "k".
merged_df = pd.merge(df_main, pivot_df, on=['Target Ratio', 'k'], how='left')

# Write the merged DataFrame to a new CSV file.
merged_df.to_csv(output_csv, index=False)

print(f'Merged CSV file saved as {output_csv}')
