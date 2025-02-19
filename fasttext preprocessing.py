import numpy as np
import matplotlib.pyplot as plt


def load_vectors(fname):
    """
    Load word vectors from a file.

    Args:
        fname (str): Path to the file containing word vectors.

    Returns:
        dict: A dictionary where keys are words and values are vectors (as numpy arrays).
    """
    try:
        with open(fname, 'r', encoding='utf-8', errors='ignore') as fin:
            n, d = map(int, fin.readline().split())
            print(f"Number of vectors: {n}, Dimension of each vector: {d}")

            data = {}
            for line in fin:
                tokens = line.rstrip().split(' ')
                word, vector = tokens[0], np.array(tokens[1:], dtype=np.float32)
                data[word] = vector
        return data
    except FileNotFoundError:
        print(f"File not found: {fname}")
        return {}
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return {}


def compute_dimension_stats(vector_array, remove_extremes=False, percentage=1.0):
    """
    Compute statistics per dimension with an option to remove top and bottom percentage of values.

    Args:
        vector_array (np.ndarray): Array of vectors.
        remove_extremes (bool): If True, removes top and bottom percentage of values before computing statistics.
        percentage (float): Percentage of max and min values to remove if remove_extremes is True.

    Returns:
        dict: Statistics (max, min, avg, range, std, percentage_range) per dimension.
    """
    if remove_extremes:
        n_remove = int(vector_array.shape[0] * (percentage / 100))
        sorted_vectors = np.sort(vector_array, axis=0)
        trimmed_vectors = sorted_vectors[n_remove:-n_remove]  # Remove top and bottom 1% values
    else:
        trimmed_vectors = vector_array

    max_per_dim = np.max(trimmed_vectors, axis=0)
    min_per_dim = np.min(trimmed_vectors, axis=0)
    avg_per_dim = np.mean(trimmed_vectors, axis=0)
    range_per_dim = max_per_dim - min_per_dim
    std_per_dim = np.std(trimmed_vectors, axis=0)

    percentage_range = np.where(max_per_dim != 0, (range_per_dim / max_per_dim) * 100, 0)

    return {
        "max": max_per_dim,
        "min": min_per_dim,
        "avg": avg_per_dim,
        "range": range_per_dim,
        "std": std_per_dim,
        "percentage_range": percentage_range,
    }


def plot_dimension_stats(dimension_stats, title_suffix="", label_extremes=False, percentage=1.0):
    """
    Plot statistics per dimension.

    Args:
        dimension_stats (dict): Statistics per dimension.
        title_suffix (str): Suffix for plot titles.
        label_extremes (bool): If True, adds a label for removed extremes.
        percentage (float): Percentage of max and min values removed if label_extremes is True.
    """
    dimensions = np.arange(len(dimension_stats["avg"]))
    label = f" (Top and Bottom {percentage}% Removed)" if label_extremes else ""

    plt.figure(figsize=(15, 12))

    # Plot averages per dimension
    plt.subplot(3, 2, 1)
    plt.plot(dimensions, dimension_stats["avg"], label='Average')
    plt.title(f"Average Per Dimension {title_suffix}{label}", fontsize=14)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    # Plot standard deviations per dimension
    plt.subplot(3, 2, 2)
    plt.plot(dimensions, dimension_stats["std"], label='Standard Deviation', color='orange')
    plt.title(f"Standard Deviation Per Dimension {title_suffix}{label}", fontsize=14)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    # Plot ranges per dimension
    plt.subplot(3, 2, 3)
    plt.plot(dimensions, dimension_stats["range"], label='Range', color='green')
    plt.title(f"Range Per Dimension {title_suffix}{label}", fontsize=14)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    # Plot percentage range per dimension
    plt.subplot(3, 2, 4)
    plt.plot(dimensions, dimension_stats["percentage_range"], label='Percentage Range', color='red')
    plt.title(f"Percentage Range Per Dimension {title_suffix}{label}", fontsize=14)
    plt.xlabel("Dimension")
    plt.ylabel("Percentage (%)")
    plt.grid(True)
    plt.legend()

    # Plot maximum values per dimension
    plt.subplot(3, 2, 5)
    plt.plot(dimensions, dimension_stats["max"], label='Maximum', color='purple')
    plt.title(f"Maximum Per Dimension {title_suffix}{label}", fontsize=14)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    # Plot minimum values per dimension
    plt.subplot(3, 2, 6)
    plt.plot(dimensions, dimension_stats["min"], label='Minimum', color='brown')
    plt.title(f"Minimum Per Dimension {title_suffix}{label}", fontsize=14)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# File path
filename = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'

# Load vectors
vectors = load_vectors(filename)

if vectors:
    # Convert to numpy array
    vector_array = np.array(list(vectors.values()))

    # Compute and plot stats for original data
    dimension_stats = compute_dimension_stats(vector_array)
    plot_dimension_stats(dimension_stats, title_suffix="(Original)")

    # Compute and plot stats with top and bottom 1% removed
    dimension_stats_trimmed = compute_dimension_stats(vector_array, remove_extremes=True, percentage=1.0)
    plot_dimension_stats(dimension_stats_trimmed, title_suffix="(Trimmed)", label_extremes=True, percentage=1.0)
