import numpy as np

def exclude_extreme_values(vector_array, top_exclude=50):
    """
    Compute the standard deviation of each dimension, excluding the top max and min values.

    Args:
        vector_array (np.ndarray): The original high-dimensional vectors.
        top_exclude (int): Number of top max and min values to exclude from each dimension.

    Returns:
        np.ndarray: Standard deviation of each dimension after exclusion.
    """
    sorted_array = np.sort(vector_array, axis=0)
    trimmed_array = sorted_array[top_exclude:-top_exclude, :]  # Exclude top and bottom values
    return np.std(trimmed_array, axis=0)

def preserve_top_dimensions(vector_array, top_dimensions=30, top_exclude=50):
    """
    Preserve the top dimensions based on standard deviation after excluding extremes.

    Args:
        vector_array (np.ndarray): The original high-dimensional vectors.
        top_dimensions (int): Number of dimensions to preserve.
        top_exclude (int): Number of top max and min values to exclude for std computation.

    Returns:
        np.ndarray: Reduced-dimensional vectors.
        np.ndarray: Indices of the selected dimensions.
    """
    print(f"Original shape: {vector_array.shape}")

    # Compute standard deviation per dimension
    std_devs = exclude_extreme_values(vector_array, top_exclude=top_exclude)
    print(f"Computed standard deviations after excluding {top_exclude} extremes per dimension.")

    # Select indices of the top dimensions based on standard deviation
    top_indices = np.argsort(std_devs)[-top_dimensions:]  # Get indices of top `top_dimensions` std-dev
    print(f"Selected top {top_dimensions} dimensions with highest std-dev.")

    # Reduce the vector array to the selected dimensions
    reduced_array = vector_array[:, top_indices]
    print(f"Reduced shape: {reduced_array.shape}")

    return reduced_array, top_indices

def load_vectors(fname):
    """
    Load word vectors from a file.

    Args:
        fname (str): Path to the file containing word vectors.

    Returns:
        dict: Dictionary of word vectors.
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

if __name__ == "__main__":
    # File paths
    training_file = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'
    testing_file = 'testing_set_vectors.npy'

    # Load training vectors
    print("\nLoading training vectors:")
    vectors = load_vectors(training_file)

    if vectors:
        # Convert to numpy array
        words = list(vectors.keys())
        vector_array = np.array(list(vectors.values()))

        # Perform custom dimension reduction on training set
        target_dimensions = 30
        reduced_vectors, top_indices = preserve_top_dimensions(vector_array, top_dimensions=target_dimensions, top_exclude=50)

        # Save reduced training vectors
        np.save("10000reduced_vectors_topStdOnly.npy", reduced_vectors)
        print("Reduced training vectors saved to 10000reduced_vectors_topStdOnly.npy")

        # Save words (optional)
        np.save("../words.npy", np.array(words))
        print("Words saved to ../words.npy")

        # Load testing vectors
        print("\nLoading testing vectors:")
        test_vectors = np.load(testing_file)
        print(f"Testing vectors shape: {test_vectors.shape}")

        # Reduce testing vectors using the same dimensions
        reduced_test_vectors = test_vectors[:, top_indices]
        print(f"Reduced testing vectors shape: {reduced_test_vectors.shape}")

        # Save reduced testing vectors
        np.save("100testing_reduced_vectors_topStdOnly.npy", reduced_test_vectors)
        print("Reduced testing vectors saved to 100testing_reduced_vectors_topStdOnly.npy")
