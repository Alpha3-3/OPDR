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
        vectors (dict): Dictionary of original high-dimensional vectors.
        target_dimension (int): Target dimension for reduction.

    Returns:
        np.ndarray: Projected vectors of lower dimension.
        list: Words corresponding to the reduced vectors.
    """
    words = list(vectors.keys())
    vector_array = np.array(list(vectors.values()))  # Shape: (n_samples, original_dimension)

    original_dimension = vector_array.shape[1]
    print(f"Original dimension: {original_dimension}, Target dimension: {target_dimension}")

    # Step 1: Generate a random matrix with entries from N(0, 1)
    random_matrix = np.random.normal(0, 1, (target_dimension, original_dimension))

    # Step 2: Scale the matrix by 1/sqrt(target_dimension)
    scaled_matrix = random_matrix / np.sqrt(target_dimension)

    # Step 3: Perform the random projection
    reduced_vectors = np.dot(vector_array, scaled_matrix.T)

    print(f"Reduced vectors shape: {reduced_vectors.shape}")

    return reduced_vectors, words


if __name__ == "__main__":
    # File path
    filename = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'

    # Load vectors
    vectors = load_vectors(filename)

    if vectors:
        # Save original vectors
        vector_array = np.array(list(vectors.values()))

        # Specify the target dimension
        target_dimension = 30  # Change this as needed

        # Reduce dimensions using random projection
        reduced_vectors, words = random_projection(vectors, target_dimension)

        # Save reduced vectors for later use
        np.save("10000reduced_vectors_randomProjection.npy", reduced_vectors)
        np.save("../words.npy", np.array(words))
        print("Reduced vectors and words saved.")
