import numpy as np
import glob
import random


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


def load_reduced_and_testing_data():
    """
    Load reduced vectors for training and testing sets.

    Returns:
        dict: Reduced training vectors from different methods.
        dict: Reduced testing vectors from different methods.
    """
    reduced_train_vectors = {}
    reduced_test_vectors = {}

    # Load all reduced training set vectors
    for filepath in glob.glob("10000reduced_vectors_*.npy"):
        method_name = filepath.split("_")[-1].split(".")[0]  # Extract method name from file
        reduced_train_vectors[method_name.upper()] = np.load(filepath)

    # Load all reduced testing set vectors
    for filepath in glob.glob("100testing_reduced_vectors_*.npy"):
        method_name = filepath.split("_")[-1].split(".")[0]  # Extract method name from file
        reduced_test_vectors[method_name.upper()] = np.load(filepath)

    return reduced_train_vectors, reduced_test_vectors


def check_testing_accuracy(original_vectors, reduced_train_vectors, reduced_test_vectors, k=10):
    """
    Check the accuracy of dimensionality reduction methods using the testing set.

    Args:
        original_vectors (np.ndarray): Original high-dimensional vectors.
        reduced_train_vectors (dict): Reduced training vectors from different methods.
        reduced_test_vectors (dict): Reduced testing vectors from different methods.
        k (int): Number of nearest neighbors to consider.

    Returns:
        dict: Accuracy results for each method.
    """
    accuracies = {}

    for method, test_vectors in reduced_test_vectors.items():
        if method not in reduced_train_vectors:
            print(f"Warning: No matching training set for method {method}. Skipping.")
            continue

        train_vectors = reduced_train_vectors[method]

        method_accuracies = []
        for i in range(test_vectors.shape[0]):
            # Find original nearest neighbors for the testing point
            distances_original = np.linalg.norm(original_vectors - original_vectors[i], axis=1)
            original_neighbors = np.argsort(distances_original)[1:k + 1]

            # Find reduced nearest neighbors for the testing point
            distances_reduced = np.linalg.norm(train_vectors - test_vectors[i], axis=1)
            reduced_neighbors = np.argsort(distances_reduced)[:k]

            # Calculate accuracy for this point
            common_neighbors = set(original_neighbors).intersection(reduced_neighbors)
            accuracy = len(common_neighbors) / k * 100
            method_accuracies.append(accuracy)

        # Average accuracy for the current method
        avg_accuracy = np.mean(method_accuracies)
        accuracies[method] = avg_accuracy

    return accuracies


if __name__ == "__main__":
    # File path for original high-dimensional vectors
    file_path = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'

    # Load original high-dimensional vectors
    vectors = load_vectors(file_path)
    original_vectors = np.array(list(vectors.values()))

    # Load reduced vectors for training and testing sets
    reduced_train_vectors, reduced_test_vectors = load_reduced_and_testing_data()

    # Check accuracy using the testing set
    accuracies = check_testing_accuracy(original_vectors, reduced_train_vectors, reduced_test_vectors, k=10)

    # Save accuracy results
    with open("../testing_accuracy_results.txt", "w") as f:
        for method, accuracy in accuracies.items():
            result = f"Accuracy for {method}: {accuracy:.2f}%"
            print(result)
            f.write(result + "\n")
