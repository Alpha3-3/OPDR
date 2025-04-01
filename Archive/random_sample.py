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
                word, vector = tokens[0], np.array(tokens[1:], dtype=np.float32)
                data[word] = vector
        return data
    except FileNotFoundError:
        print(f"File not found: {fname}")
        return {}
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return {}

# Function to sample 10000 random points from the dataset
def sample_vectors(data, sample_size=10000):
    """
    Randomly sample a subset of vectors from the dataset.

    Args:
        data (dict): Dictionary of word vectors.
        sample_size (int): Number of points to sample.

    Returns:
        dict: Sampled subset of word vectors.
    """
    if sample_size > len(data):
        print("Sample size is larger than the dataset size. Returning the entire dataset.")
        return data

    sampled_keys = random.sample(list(data.keys()), sample_size)
    sampled_data = {key: data[key] for key in sampled_keys}
    return sampled_data

# Load your dataset
file_path = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M.vec'  # Replace with the actual file path
vectors = load_vectors(file_path)

# Sample 10,000 random points
sampled_vectors = sample_vectors(vectors, sample_size=10000)



def save_vectors(vecs, fname):
    """
    Save word vectors to a file.

    Args:
        vecs (dict): Dictionary of word vectors.
        fname (str): Path to the file where vectors will be saved.

    Returns:
        None
    """
    try:
        with open(fname, 'w', encoding='utf-8') as fout:
            # Write the header line: number of vectors and dimensions
            fout.write(f"{len(vecs)} {len(next(iter(vecs.values())))}\n")

            # Write each word and its vector
            for word, vector in vecs.items():
                vec_str = ' '.join(map(str, vector))
                fout.write(f"{word} {vec_str}\n")
        print(f"Sampled vectors saved to {fname}.")
    except Exception as e:
        print(f"Error saving vectors: {e}")

# Save the sampled vectors to a .vec file
output_file = r'D:\My notes\UW\HPDIC Lab\OPDR\wiki-news-300d-1M\wiki-news-300d-1M-sampled.vec'
save_vectors(sampled_vectors, output_file)

print(f"Sampled {len(sampled_vectors)} vectors.")
