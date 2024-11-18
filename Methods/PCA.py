import numpy as np
from sklearn.decomposition import PCA


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


def reduce_dimension_pca(vectors, target_dimension):
    """
    Reduce the dimensionality of the word vectors using PCA.

    Args:
        vectors (dict): Dictionary of word vectors.
        target_dimension (int): Target dimensionality after PCA reduction.

    Returns:
        np.ndarray: Word vectors reduced to the target dimension.
        list: Words corresponding to the reduced vectors.
        PCA: The PCA object after fitting.
    """
    words = list(vectors.keys())
    vector_array = np.array(list(vectors.values()))

    print(f"Original dimension: {vector_array.shape[1]}")
    print(f"Target dimension: {target_dimension}")

    # Apply PCA
    pca = PCA(n_components=target_dimension)
    reduced_vectors = pca.fit_transform(vector_array)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

    return reduced_vectors, words, pca


if __name__ == "__main__":
    # File path for training set
    training_filename = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'

    # Load training vectors
    training_vectors = load_vectors(training_filename)

    if training_vectors:
        # Convert training vectors to numpy array
        training_vector_array = np.array(list(training_vectors.values()))

        # Specify the target dimension
        target_dimension = 30

        # Reduce dimensions using PCA on training set
        reduced_training_vectors, words, pca = reduce_dimension_pca(training_vectors, target_dimension)

        # Save reduced training vectors and words for later use
        np.save("10000reduced_vectors_pca.npy", reduced_training_vectors)
        np.save("../words.npy", np.array(words))
        print("Reduced training vectors and words saved.")

        # Load testing set vectors
        test_vector_array = np.load('testing_set_vectors.npy', allow_pickle=True)

        # Handle scalar array case
        if test_vector_array.shape == ():
            print("test_vector_array is a scalar. Extracting the contained object.")
            test_vector_array = test_vector_array.item()
            print(f"Extracted object type: {type(test_vector_array)}")

            # Process based on the extracted object's type
            if isinstance(test_vector_array, dict):
                print(f"Testing set is a dictionary with {len(test_vector_array)} items.")
                test_vector_array = np.array(list(test_vector_array.values()))
            elif isinstance(test_vector_array, list):
                print(f"Testing set is a list with {len(test_vector_array)} elements.")
                test_vector_array = np.vstack(test_vector_array)
            elif isinstance(test_vector_array, np.ndarray):
                print(f"Testing set is already a NumPy array with shape: {test_vector_array.shape}")
            else:
                print("Error: Unsupported testing set format.")
                exit(1)

        # Ensure the test vectors have the same original dimensionality
        if test_vector_array.ndim != 2:
            print("Error: test_vector_array is not a 2D array.")
            exit(1)

        if test_vector_array.shape[1] != training_vector_array.shape[1]:
            print(f"Error: The testing vectors have dimensionality {test_vector_array.shape[1]}, "
                  f"which does not match the training vectors' dimensionality {training_vector_array.shape[1]}.")
            exit(1)
        else:
            # Apply PCA transformation to the testing set
            reduced_test_vectors = pca.transform(test_vector_array)
            np.save("100testing_reduced_vectors_pca.npy", reduced_test_vectors)
            print("Reduced testing vectors saved.")
