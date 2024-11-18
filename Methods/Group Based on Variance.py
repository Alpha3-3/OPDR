import numpy as np
import json

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
    """
    min_values = np.min(vectors, axis=0)
    adjustment = np.abs(np.minimum(0, min_values))  # Only add if min_value < 0
    return vectors + adjustment

def group_dimensions(vectors, target_dimension, save_group_file=None):
    """
    Group dimensions based on variance and balance them across groups.

    Args:
        vectors (np.ndarray): 2D numpy array of shape (m, n) where m is the number of vectors and n is the dimensionality.
        target_dimension (int): Desired number of dimensions after grouping.
        save_group_file (str): Optional path to save the grouping information.

    Returns:
        np.ndarray: Transformed vectors with reduced dimensions.
        list: Grouping information.
    """
    variances = np.var(vectors, axis=0)
    sorted_indices = np.argsort(-variances)  # Sort in descending order of variance
    sorted_variances = variances[sorted_indices]

    groups = [[] for _ in range(target_dimension)]
    group_variances = np.zeros(target_dimension)

    for idx, var in zip(sorted_indices, sorted_variances):
        min_group = np.argmin(group_variances)
        groups[min_group].append(int(idx))  # Convert to standard Python int
        group_variances[min_group] += var

    new_vectors = np.zeros((vectors.shape[0], target_dimension))
    for i, group in enumerate(groups):
        new_vectors[:, i] = np.sqrt(np.sum(np.square(vectors[:, group]), axis=1))

    # Optionally save grouping information
    if save_group_file:
        # Ensure all indices are Python int for JSON serialization
        groups_json_serializable = [[int(idx) for idx in group] for group in groups]
        with open(save_group_file, "w") as f:
            json.dump(groups_json_serializable, f)
        print(f"Saved grouping information to {save_group_file}")

    return new_vectors, groups

def apply_grouping_to_testing(vectors, groups):
    """
    Apply pre-computed grouping to the testing vectors.

    Args:
        vectors (np.ndarray): Testing set vectors.
        groups (list): Pre-computed grouping information.

    Returns:
        np.ndarray: Transformed testing set vectors with reduced dimensions.
    """
    target_dimension = len(groups)
    new_vectors = np.zeros((vectors.shape[0], target_dimension))
    for i, group in enumerate(groups):
        new_vectors[:, i] = np.sqrt(np.sum(np.square(vectors[:, group]), axis=1))
    return new_vectors

if __name__ == "__main__":
    # File paths
    training_file = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'
    testing_file = "testing_set_vectors.npy"

    # Load training vectors
    training_vectors = load_vectors(training_file)

    if training_vectors:
        words = list(training_vectors.keys())
        training_array = np.array(list(training_vectors.values()))

        # Make non-negative
        training_array = make_non_negative(training_array)

        # Specify target dimension
        target_dimension = 30

        # Group dimensions and save grouping info
        reduced_training_vectors, groups = group_dimensions(
            training_array,
            target_dimension,
            save_group_file="dimension_groups.json"
        )

        # Save reduced training vectors
        np.save("10000reduced_vectors_groupBasedOnVar.npy", reduced_training_vectors)
        np.save("../words.npy", np.array(words))
        print("Reduced training vectors and words saved.")

        # Load testing vectors
        testing_array = np.load(testing_file, allow_pickle=True)

        # Handle scalar array case
        if testing_array.shape == ():
            print("testing_array is a scalar. Extracting the contained object.")
            testing_array = testing_array.item()
            print(f"Extracted object type: {type(testing_array)}")

            # Process based on the extracted object's type
            if isinstance(testing_array, dict):
                print(f"Testing set is a dictionary with {len(testing_array)} items.")
                testing_array = np.array(list(testing_array.values()))
            elif isinstance(testing_array, list):
                print(f"Testing set is a list with {len(testing_array)} elements.")
                testing_array = np.vstack(testing_array)
            elif isinstance(testing_array, np.ndarray):
                print(f"Testing set is already a NumPy array with shape: {testing_array.shape}")
            else:
                print("Error: Unsupported testing set format.")
                exit(1)

        # Ensure the test vectors have the expected dimensions
        if testing_array.ndim != 2:
            print("Error: testing_array is not a 2D array.")
            exit(1)

        # Make non-negative
        testing_array = make_non_negative(testing_array)

        # Apply grouping to testing set
        reduced_testing_vectors = apply_grouping_to_testing(testing_array, groups)

        # Save reduced testing vectors
        np.save("100testing_reduced_vectors_groupBasedOnVar.npy", reduced_testing_vectors)
        print("Reduced testing vectors saved.")
