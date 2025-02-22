import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import coolwarm

def ensure_difference_column(df):
    """
    Ensure that the DataFrame has a 'Difference' column.
    If it doesn't, compute it as the difference between
    'DW-PMAD Accuracy' and 'PCA Accuracy'.
    """
    if 'Difference' not in df.columns:
        # Calculate the difference as DW-PMAD Accuracy minus PCA Accuracy.
        df['Difference'] = df['DW-PMAD Accuracy'] - df['PCA Accuracy']
    return df

# Load the data
file_path = 'parameter_sweep_results_isolet.csv'
df = pd.read_csv(file_path)
df = ensure_difference_column(df)

# Extract necessary columns
x = df['Target Ratio']  # X-axis
y = df['b']             # Y-axis
z = df['alpha']         # Z-axis
diff = df['Difference'] # Color intensity
k = df['k']             # Marker shape

# Define marker styles for different k values
marker_styles = {3: 's', 6: '^', 9: 'o', 12: 'd'}  # Square, triangle, circle, diamond

# Normalize colors based on the difference value with better contrast
norm = TwoSlopeNorm(vmin=min(diff), vcenter=0, vmax=max(diff))
cmap = coolwarm
colors = [cmap(norm(d)) for d in diff]

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Introduce small jitter to separate overlapping points
jitter_strength = 0.02
x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))
y_jittered = y + np.random.uniform(-jitter_strength, jitter_strength, size=len(y))
z_jittered = z + np.random.uniform(-jitter_strength, jitter_strength, size=len(z))

# Scatter plot with different markers and colors
for i in range(len(df)):
    ax.scatter(x_jittered[i], y_jittered[i], z_jittered[i],
               color=colors[i],
               marker=marker_styles.get(k[i], 'o'),
               s=50)

# Labels and title
ax.set_xlabel('Target Ratio')
ax.set_ylabel('b')
ax.set_zlabel('Alpha')
ax.set_title('DW-PMAD vs. PCA Performance_Isolet')

# Add legend for marker styles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker=marker, color='w', label=f'k={key}', markersize=10, markerfacecolor='black')
    for key, marker in marker_styles.items()
]
ax.legend(handles=legend_elements, loc='upper right')

# Show the plot
plt.show()
