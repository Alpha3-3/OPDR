import numpy as np

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

def make_non_negative(vectors):
    """
    Adjust the values in each dimension of the vectors to be non-negative.

    Args:
        vectors (np.ndarray): 2D numpy array of shape (m, n) where m is the
                              number of vectors and n is the dimensionality.

    Returns:
        np.ndarray: Adjusted vectors with all values non-negative.
    """
    # Compute the minimum value for each dimension
    min_values = np.min(vectors, axis=0)

    # Adjust vectors by adding the absolute value of minimum values (if negative)
    adjustment = np.abs(np.minimum(0, min_values))  # Only add if min_value < 0
    adjusted_vectors = vectors + adjustment

    return adjusted_vectors

def compute_adjusted_variances(vectors):
    """
    Compute adjusted variances by removing top 50 max and min values.

    Args:
        vectors (np.ndarray): Array of vectors.

    Returns:
        np.ndarray: Adjusted variances of each dimension.
    """
    variances = np.var(vectors, axis=0)
    sorted_variances = np.sort(variances)
    trimmed_variances = sorted_variances[50:-50]  # Remove top 50 max and min values
    return trimmed_variances

def group_dimensions(vectors, target_dimension):
    """
    Group dimensions such that the variance of the square sum of each group is balanced.

    Args:
        vectors (np.ndarray): 2D numpy array where each row is a vector, and each column is a dimension.
        target_dimension (int): Desired number of dimensions after grouping.

    Returns:
        np.ndarray: Transformed vectors with reduced dimensions.
    """
    # Calculate variance of each dimension
    variances = np.var(vectors, axis=0)

    # Sort dimensions by variance in descending order
    sorted_indices = np.argsort(-variances)  # Descending order
    sorted_variances = variances[sorted_indices]

    # Initialize groups and their variance sums
    groups = [[] for _ in range(target_dimension)]
    group_variances = np.zeros(target_dimension)

    # Distribute dimensions to groups, prioritizing variance balance
    for idx, var in zip(sorted_indices, sorted_variances):
        # Find the group with the smallest total variance sum
        min_group = np.argmin(group_variances)
        groups[min_group].append(idx)
        group_variances[min_group] += var  # Update the variance sum of the chosen group

    # Create the new grouped dimensions
    new_vectors = np.zeros((vectors.shape[0], target_dimension))
    for i, group in enumerate(groups):
        # Compute square sum for each group to form new dimension
        new_vectors[:, i] = np.sqrt(np.sum(np.square(vectors[:, group]), axis=1))

    return new_vectors
if __name__ == "__main__":
    # File path
    filename = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'

    # Load vectors
    vectors = load_vectors(filename)

    if vectors:
        words = list(vectors.keys())
        vector_array = np.array(list(vectors.values()))

        # Save original vectors

        # Specify the target dimension
        target_dimension = 30  # Change this as needed

        vector_array = make_non_negative(vector_array)

        # Group dimensions and reduce
        reduced_vectors = group_dimensions(vector_array, target_dimension)

        # Save reduced vectors for later use
        np.save("10000reduced_vectors_groupBasedOnVar.npy", reduced_vectors)
        np.save("../words.npy", np.array(words))
        print("Reduced vectors and words saved.")
