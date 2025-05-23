import numpy as np
import random
import os
from tqdm import tqdm

# Set random seed for reproducibility for the entire train/test split
# It's good practice to set both if you are using both 'random' and 'numpy.random'
# np.random.permutation will be the key random operation for splitting.
fixed_seed = 1 # Using a common seed, you can change this
random.seed(fixed_seed)
np.random.seed(fixed_seed)

# Path to the .vec file
# Make sure this path is correct for your system.
# Using a raw string (r'') is good for Windows paths.
vec_file_path = r'D:\My notes\UW\HPDIC Lab\OPDR\datasets\wiki-news-300d-1M.vec'

# Define the number of vectors you want for training and testing
num_training_vectors = 300  # You can change this value as needed
num_testing_vectors = 300   # You can change this value as needed

# Ensure the file exists
if not os.path.exists(vec_file_path):
    raise FileNotFoundError(f"The file {vec_file_path} does not exist. Please check the path.")

# Initialize an empty list to store the vectors
all_vectors_list = []
words_list = [] # Optionally store words if needed later, not strictly required for splitting vectors

print(f"Reading vectors from {vec_file_path}...")
# Reading the .vec file, skipping the first line since it typically contains metadata (count, dim)
try:
    with open(vec_file_path, 'r', encoding='utf-8') as file:
        try:
            header = next(file)  # Skip the header line
            print(f"Skipped header: {header.strip()}")
            num_total_vectors_in_file, dim = map(int, header.split())
            print(f"File reports {num_total_vectors_in_file} vectors of dimension {dim}.")
        except StopIteration:
            raise ValueError("File is empty or contains only a header.")
        except ValueError:
            print("Warning: Could not parse header for count and dimension. Proceeding line by line.")
            num_total_vectors_in_file = None # Unknown

        # Process each line after the header
        # Using tqdm with total if known, otherwise it will run without a total.
        for line in tqdm(file, desc="Reading vectors", total=num_total_vectors_in_file):
            parts = line.strip().split()
            if len(parts) < 2: # Expecting at least a word and one dimension
                # print(f"Warning: Skipping malformed line: '{line.strip()}'") # Can be very verbose
                continue
            # word = parts[0] # The word itself
            try:
                vector_components = np.array(parts[1:], dtype=float)
                # Optionally, you can check if vector_components.shape[0] matches 'dim' from header
                all_vectors_list.append(vector_components)
                # words_list.append(word) # Uncomment if you need to keep track of words
            except ValueError:
                # print(f"Warning: Could not convert vector components to float for word '{parts[0]}'. Skipping line.") # Can be very verbose
                continue

except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    raise

if not all_vectors_list:
    raise ValueError("No valid vectors were read from the file. Please check the file content and format.")

# Convert list to a NumPy array
all_vectors = np.array(all_vectors_list)

print(f"\nSuccessfully read and processed {all_vectors.shape[0]} vectors.")
if all_vectors.shape[0] > 0:
    print(f"Dimension of each vector: {all_vectors.shape[1]}")
else:
    # This case should ideally be caught by `if not all_vectors_list`
    raise ValueError("Vector array is empty after processing the file.")


# Check if the requested number of vectors exceeds the available data
total_requested_vectors = num_training_vectors + num_testing_vectors
if total_requested_vectors > all_vectors.shape[0]:
    raise ValueError(
        f"The sum of training ({num_training_vectors}) and testing ({num_testing_vectors}) vectors "
        f"({total_requested_vectors}) exceeds the total number of available valid vectors "
        f"({all_vectors.shape[0]})."
    )

# Generate a permutation of all available indices
all_indices = np.arange(all_vectors.shape[0])
shuffled_indices = np.random.permutation(all_indices)

# Select indices for the training set
training_indices = shuffled_indices[:num_training_vectors]

# Select indices for the testing set (ensuring no overlap)
testing_indices = shuffled_indices[num_training_vectors : num_training_vectors + num_testing_vectors]

# Retrieve the training vectors
training_set_vectors = all_vectors[training_indices]

# Save the training vectors to a .npy file
training_npy_file_path = 'training_vectors_300_Fasttext.npy' # Changed filename
np.save(training_npy_file_path, training_set_vectors)
print(f"\nSaved {training_set_vectors.shape[0]} training vectors with dimension {training_set_vectors.shape[1]} to '{training_npy_file_path}'.")

# Retrieve the testing vectors
testing_set_vectors = all_vectors[testing_indices]

# Save the testing vectors to a .npy file
testing_npy_file_path = 'testing_vectors_300_Fasttext.npy' # Kept original name for testing as per draft, but you might want to make it consistent
# np.save(testing_npy_file_path, testing_set_vectors)
print(f"Saved {testing_set_vectors.shape[0]} testing vectors with dimension {testing_set_vectors.shape[1]} to '{testing_npy_file_path}'.")

# --- Verification of no overlap (optional) ---
# Check if any index is shared (which it shouldn't be by design)
shared_indices = np.intersect1d(training_indices, testing_indices)
if len(shared_indices) == 0:
    print("\nVerification successful: No overlap between training and testing sets based on indices.")
else:
    # This should not happen with the current logic
    print(f"\nVerification FAILED: Found {len(shared_indices)} overlapping indices between training and testing sets.")

print(f"\nTotal vectors in training set: {len(training_set_vectors)}")
print(f"Total vectors in testing set: {len(testing_set_vectors)}")