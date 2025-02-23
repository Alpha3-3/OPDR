import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import coolwarm
from matplotlib.lines import Line2D

def ensure_difference_column(df):
    """
    Ensure that the DataFrame has a 'Difference' column.
    If not, compute it as the difference between
    'DW-PMAD Accuracy' and 'PCA Accuracy'.
    """
    if 'Difference' not in df.columns:
        df['Difference'] = df['DW-PMAD Accuracy'] - df['PCA Accuracy']
    return df

def plot_3d_scatter(file_path, plot_title):
    # Load and prepare the data
    df = pd.read_csv(file_path)
    df = ensure_difference_column(df)

    # Convert columns to numpy arrays for easier indexing
    x = df['Target Ratio'].values
    y = df['b'].values
    z = df['alpha'].values
    diff = df['Difference'].values
    k = df['k'].values

    # Define marker styles for different k values
    marker_styles = {3: 's', 6: '^', 9: 'o', 12: 'd'}  # square, triangle, circle, diamond

    # Normalize colors based on 'Difference'
    norm = TwoSlopeNorm(vmin=diff.min(), vcenter=0, vmax=diff.max())
    cmap = coolwarm
    colors = [cmap(norm(val)) for val in diff]

    # Add jitter to separate overlapping points
    jitter_strength = 0.02
    x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=x.shape)
    y_jittered = y + np.random.uniform(-jitter_strength, jitter_strength, size=y.shape)
    z_jittered = z + np.random.uniform(-jitter_strength, jitter_strength, size=z.shape)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point with the corresponding marker and color
    for i in range(len(x)):
        ax.scatter(x_jittered[i], y_jittered[i], z_jittered[i],
                   color=colors[i],
                   marker=marker_styles.get(k[i], 'o'),
                   s=50)

    # Set labels and title
    ax.set_xlabel('Target Ratio')
    ax.set_ylabel('b')
    ax.set_zlabel('Alpha')
    ax.set_title(plot_title)

    # Add a legend for marker styles
    legend_elements = [
        Line2D([0], [0], marker=marker, color='w', label=f'k={key}',
               markersize=10, markerfacecolor='black')
        for key, marker in marker_styles.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()

# Plot for each dataset
plot_3d_scatter('Higgs/parameter_sweep_results_Higgs.csv', 'DW-PMAD vs. PCA Performance_Higgs')
plot_3d_scatter('Fasttext/parameter_sweep_results_fasttext.csv', 'DW-PMAD vs. PCA Performance_Fasttext')
plot_3d_scatter('CIFAR-10/parameter_sweep_results_CIFAR-10.csv', 'DW-PMAD vs. PCA Performance_CIFAR-10')
plot_3d_scatter('Isolet/parameter_sweep_results_isolet.csv', 'DW-PMAD vs. PCA Performance_Isolet')
