import numpy as np
from umap import UMAP
import time


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


def reduce_dimension_umap(vectors, target_dimension):
    """
    Reduce the dimensionality of the word vectors using UMAP.

    Args:
        vectors (dict): Dictionary of word vectors.
        target_dimension (int): Target dimensionality after UMAP reduction.

    Returns:
        np.ndarray: Word vectors reduced to the target dimension.
        list: Words corresponding to the reduced vectors.
    """
    words = list(vectors.keys())
    vector_array = np.array(list(vectors.values()))

    print(f"Original dimension: {vector_array.shape[1]}")
    print(f"Target dimension: {target_dimension}")
    print(f"Number of vectors: {vector_array.shape[0]}")

    # Apply UMAP
    umap = UMAP(n_components=target_dimension, metric='euclidean', n_jobs=-1)
    start_time = time.time()
    reduced_vectors = umap.fit_transform(vector_array)
    print(f"UMAP completed in {time.time() - start_time:.2f} seconds.")

    return reduced_vectors, words


if __name__ == "__main__":
    # File path
    filename = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'

    # Load vectors
    vectors = load_vectors(filename)

    if vectors:
        # Specify the target dimension
        target_dimension = 30  # Change this as needed

        # Reduce dimensions using UMAP
        reduced_vectors, words = reduce_dimension_umap(vectors, target_dimension)

        # Save reduced vectors for later use
        np.save("10000reduced_vectors_umap.npy", reduced_vectors)
        np.save("../words.npy", np.array(words))
        print("Reduced vectors and words saved.")
