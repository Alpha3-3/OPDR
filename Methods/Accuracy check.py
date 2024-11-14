import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import random
import glob

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


def load_reduced_data():
    """
    Load reduced vectors and words from saved files.

    Returns:
        dict: A dictionary of reduced vectors from different methods.
        np.ndarray: The original high-dimensional vectors.
        list: Words corresponding to the vectors.
    """
    reduced_vectors = {}
    for filepath in glob.glob("10000reduced_vectors_*.npy"):
        method_name = filepath.split("_")[-1].split(".")[0]  # Extract method name from file
        reduced_vectors[method_name.upper()] = np.load(filepath)

    file_path = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'  # Replace with the actual file path
    vectors = load_vectors(file_path)
    original_vectors = np.array(list(vectors.values()))
    words = np.load("../words.npy", allow_pickle=True).tolist()
    return original_vectors, reduced_vectors, words



def precompute_original_neighbors(original_vectors, k, sample_indices):
    """
    Precompute nearest neighbors for the original vectors using Euclidean distance (L2 norm).

    Args:
        original_vectors (np.ndarray): Original high-dimensional vectors.
        k (int): Number of nearest neighbors.
        sample_indices (list): Indices of sampled points.

    Returns:
        dict: Precomputed nearest neighbors for each sampled point.
    """
    neighbors = {}
    for idx in sample_indices:
        # Compute Euclidean distances to all other points
        distances = np.linalg.norm(original_vectors - original_vectors[idx], axis=1)
        # Get indices of the k nearest neighbors (excluding the point itself)
        neighbors[idx] = np.argsort(distances)[1:k + 1]  # Exclude the point itself (distance is 0)
    return neighbors


def check_reduction_accuracy(original_vectors, reduced_vectors_dict, k=10, iterations=100):
    """
    Check the accuracy of dimensionality reduction methods by comparing nearest neighbors.

    Args:
        original_vectors (np.ndarray): The original high-dimensional vectors.
        reduced_vectors_dict (dict): A dictionary of reduced vectors from different methods.
        k (int): Number of nearest neighbors to consider.
        iterations (int): Number of random points to test.

    Returns:
        dict: A dictionary with method names as keys and average accuracy as values.
    """
    n_points = original_vectors.shape[0]
    if n_points < k:
        raise ValueError("Number of points is less than k. Cannot compute nearest neighbors.")

    # Pre-generate random sample points for consistency across methods
    sample_indices = [random.randint(0, n_points - 1) for _ in range(iterations)]

    # Precompute original nearest neighbors
    original_neighbors = precompute_original_neighbors(original_vectors, k, sample_indices)

    # Accuracy results for each method
    accuracies = {method: [] for method in reduced_vectors_dict.keys()}

    for method, reduced_vectors in reduced_vectors_dict.items():
        for idx in sample_indices:
            # Compute Euclidean distances for the current sampled point
            distances = np.linalg.norm(reduced_vectors - reduced_vectors[idx], axis=1)

            # Compute nearest neighbors in the reduced space
            reduced_neighbors = np.argsort(distances)[1:k + 1]  # Exclude the point itself

            # Compare neighbors
            common_neighbors = set(original_neighbors[idx]).intersection(set(reduced_neighbors))
            accuracy = len(common_neighbors) / k * 100  # Accuracy as a percentage
            accuracies[method].append(accuracy)

    # Compute average accuracy for each method
    avg_accuracies = {method: np.mean(acc) for method, acc in accuracies.items()}
    return avg_accuracies

if __name__ == "__main__":
    # Load data
    original_vectors, reduced_vectors_dict, _ = load_reduced_data()

    # Check accuracy for all methods
    accuracies = check_reduction_accuracy(original_vectors, reduced_vectors_dict, k=10, iterations=100)

    # Print and save results
    with open("../accuracy_results.txt", "w") as f:
        for method, accuracy in accuracies.items():
            result = f"Average accuracy of {method}: {accuracy:.2f}%"
            print(result)
            f.write(result + "\n")
