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

def random_projection(vectors, target_dimension):
    """
    Reduce the dimensionality of vectors using random projection.

    Args:
        vectors (np.ndarray): Original high-dimensional vectors.
        target_dimension (int): Target dimension for reduction.

    Returns:
        np.ndarray: Projected vectors of lower dimension.
        np.ndarray: Random projection matrix used.
    """
    original_dimension = vectors.shape[1]
    print(f"Original dimension: {original_dimension}, Target dimension: {target_dimension}")

    # Step 1: Generate a random matrix with entries from N(0, 1)
    random_matrix = np.random.normal(0, 1, (target_dimension, original_dimension))

    # Step 2: Scale the matrix by 1/sqrt(target_dimension)
    scaled_matrix = random_matrix / np.sqrt(target_dimension)

    # Step 3: Perform the random projection
    reduced_vectors = np.dot(vectors, scaled_matrix.T)

    print(f"Reduced vectors shape: {reduced_vectors.shape}")

    return reduced_vectors, scaled_matrix

def apply_random_projection_to_test_set(test_vectors, projection_matrix):
    """
    Apply the same random projection matrix to the testing set.

    Args:
        test_vectors (np.ndarray): Original testing vectors.
        projection_matrix (np.ndarray): Random projection matrix used for training set.

    Returns:
        np.ndarray: Projected testing vectors of lower dimension.
    """
    # Perform the random projection using the same projection matrix
    reduced_test_vectors = np.dot(test_vectors, projection_matrix.T)
    print(f"Reduced testing vectors shape: {reduced_test_vectors.shape}")
    return reduced_test_vectors

if __name__ == "__main__":
    # File paths
    training_file = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'
    testing_file = 'testing_set_vectors.npy'

    # Load training vectors
    training_vectors_dict = load_vectors(training_file)

    if training_vectors_dict:
        words = list(training_vectors_dict.keys())
        training_vectors = np.array(list(training_vectors_dict.values()))

        # Specify the target dimension
        target_dimension = 150  # Adjust as needed

        # Reduce dimensions using random projection
        reduced_training_vectors, projection_matrix = random_projection(training_vectors, target_dimension)

        # Save reduced training vectors
        np.save("10000reduced_vectors_randomProjection.npy", reduced_training_vectors)
        print("Reduced training vectors saved to 10000reduced_vectors_randomProjection.npy")

        # Save words (optional)
        np.save("../words.npy", np.array(words))

        # Load testing set
        test_vectors = np.load(testing_file)
        print(f"Testing set shape: {test_vectors.shape}")

        # Apply the same random projection to the testing set
        reduced_testing_vectors = apply_random_projection_to_test_set(test_vectors, projection_matrix)

        # Save reduced testing vectors
        np.save("100testing_reduced_vectors_randomProjection.npy", reduced_testing_vectors)
        print("Reduced testing vectors saved to 100testing_reduced_vectors_randomProjection.npy")
