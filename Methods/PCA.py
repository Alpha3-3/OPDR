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
        PCA: Trained PCA object.
        np.ndarray: Word vectors reduced to the target dimension.
    """
    vector_array = np.array(list(vectors.values()))
    print(f"Original dimension: {vector_array.shape[1]}")
    print(f"Target dimension: {target_dimension}")

    # Apply PCA
    pca = PCA(n_components=target_dimension)
    reduced_vectors = pca.fit_transform(vector_array)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

    return pca, reduced_vectors


def apply_pca_to_test_set(test_vectors, pca):
    """
    Apply PCA transformation learned from the training set to the testing set.

    Args:
        test_vectors (np.ndarray): Test vectors to transform.
        pca (PCA): Trained PCA object.

    Returns:
        np.ndarray: Dimension-reduced test vectors.
    """
    return pca.transform(test_vectors)


if __name__ == "__main__":
    # File paths
    training_file = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'
    testing_file = 'testing_set_vectors.npy'

    # Load training vectors
    training_vectors = load_vectors(training_file)
    if training_vectors:
        # Reduce training set dimensions
        target_dimension = 30  # Adjust target dimension if necessary
        pca, reduced_training_vectors = reduce_dimension_pca(training_vectors, target_dimension)

        # Save reduced training set
        np.save("10000reduced_vectors_pca.npy", reduced_training_vectors)
        print("Reduced training vectors saved to 10000reduced_vectors_pca.npy")

        # Load testing set
        test_vectors = np.load(testing_file)
        print(f"Testing set shape: {test_vectors.shape}")

        # Apply PCA transformation to testing set
        reduced_testing_vectors = apply_pca_to_test_set(test_vectors, pca)

        # Save reduced testing set
        np.save("100testing_reduced_vectors_pca.npy", reduced_testing_vectors)
        print("Reduced testing vectors saved to 100testing_reduced_vectors_pca.npy")
