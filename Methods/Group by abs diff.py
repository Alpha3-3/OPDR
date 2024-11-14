import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

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
            for line in tqdm(fin, total=n, desc="Loading vectors", unit="vec"):
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

def reduce_dimensions_by_grouping(vectors, num_groups):
    """
    Reduce dimensions by grouping based on average absolute differences, removing top 1% max and min differences.

    Args:
        vectors (np.ndarray): 2D numpy array where each row is a vector, and each column is a dimension.
        num_groups (int): Desired number of groups (new dimensions).

    Returns:
        np.ndarray: Reduced vectors with the new grouped dimensions.
    """
    m, n = vectors.shape
    avg_abs_diffs = np.zeros(n)  # To store average absolute differences for each dimension

    # Calculate average absolute differences with top 1% filtering
    for i in tqdm(range(n), desc="Processing dimensions", unit="dim"):
        differences = []
        for j in range(m):
            differences.extend(np.abs(vectors[j, i] - vectors[:, i]))
        differences = np.array(differences)
        # Remove top 1% max and min
        lower_percentile = np.percentile(differences, 1)
        upper_percentile = np.percentile(differences, 99)
        filtered_differences = differences[(differences >= lower_percentile) & (differences <= upper_percentile)]

        # Calculate the average of filtered differences
        avg_abs_diffs[i] = np.mean(filtered_differences)

    # Sort dimensions by average absolute differences
    sorted_indices = np.argsort(-avg_abs_diffs)
    sorted_averages = avg_abs_diffs[sorted_indices]

    # Initialize groups
    groups = [[] for _ in range(num_groups)]
    group_sums = np.zeros(num_groups)

    # Greedy grouping to balance square sums
    for avg, index in zip(sorted_averages, sorted_indices):
        min_group = np.argmin(group_sums)
        groups[min_group].append(index)
        group_sums[min_group] += avg**2

    # Calculate the new dimension values as square root of square sums
    new_dimension_values = np.sqrt(group_sums)

    # Create reduced vectors based on groups
    reduced_vectors = np.zeros((m, num_groups))
    for i, group in enumerate(groups):
        for dim in group:
            reduced_vectors[:, i] += vectors[:, dim]
        reduced_vectors[:, i] *= new_dimension_values[i] / np.sqrt(group_sums[i])

    return reduced_vectors, groups, new_dimension_values

if __name__ == "__main__":
    # File path for word vectors
    filename = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'

    # Load word vectors
    print("\nLoading word vectors:")
    vectors = load_vectors(filename)

    if vectors:
        vector_array = np.array(list(vectors.values()))
        words = list(vectors.keys())

        # Specify the number of groups (new dimensions)
        num_groups = 30  # Modify this as needed

        # Dimension reduction by grouping
        print("\nReducing dimensions using grouping:")
        reduced_vectors, groups, new_dimension_values = reduce_dimensions_by_grouping(vector_array, num_groups)

        print(f"New dimension values: {new_dimension_values}")
        print(f"Groups (indices of original dimensions): {groups}")

        # Save reduced vectors for later use
        np.save("10000reduced_vectors_groupByAbsDiff.npy", reduced_vectors)
        np.save("../words.npy", np.array(words))
        print("\nReduction completed. Reduced vectors saved to files.")
