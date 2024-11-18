import numpy as np
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
                vector = np.array(tokens[1:], dtype=np.float32)
                data[len(data)] = vector  # Use a simple numeric key
        return data
    except FileNotFoundError:
        print(f"File not found: {fname}")
        return {}
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return {}

def select_and_save_vectors(vectors, num_points=100, output_file='test_vectors.npy'):
    """
    Select a specified number of vectors randomly from the vectors dictionary and save them to a .npy file.

    Args:
        vectors (dict): Dictionary of word vectors.
        num_points (int): Number of vectors to select.
        output_file (str): Output .npy file path.
    """
    # Ensure the number of points doesn't exceed the available vectors
    num_points = min(num_points, len(vectors))
    print(f"Selecting {num_points} vectors out of {len(vectors)} available.")

    # Randomly select keys
    selected_keys = random.sample(list(vectors.keys()), num_points)

    # Extract the corresponding vectors
    selected_vectors = [vectors[key] for key in selected_keys]  # Only preserve vectors

    # Save the selected vectors as a .npy file
    np.save(output_file, selected_vectors)
    print(f"Saved {len(selected_vectors)} vectors to {output_file}")

if __name__ == "__main__":
    # File path to your vector file
    filename = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M.vec'

    # Load vectors
    vectors = load_vectors(filename)

    # Select and save 100 vectors
    select_and_save_vectors(vectors, num_points=100, output_file='testing_set_vectors.npy')
