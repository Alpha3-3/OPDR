import numpy as np
# import random # Still imported, but np.random will be preferred for permutation
import os
from tqdm import tqdm

# --- Configuration ---
# Set a single random seed for reproducibility of the entire train/test split
GLOBAL_SEED = 1
np.random.seed(GLOBAL_SEED)
# random.seed(GLOBAL_SEED) # Set if other 'random' module operations are critical elsewhere

# List of file paths to include (ARCENE dataset parts)
file_paths = [
    'ARCENE/arcene_train.data',
    'ARCENE/arcene_valid.data',
    'ARCENE/arcene_test.data'
]

# Define sample sizes for training and testing
num_training_samples = 600
num_testing_samples = 300

# Sparsity filtering configuration
SPARSITY_THRESHOLD = 0.95  # Remove columns with more than this proportion of zeros
MIN_FEATURES_TO_KEEP = 50  # Try to keep at least this many features

# --- Data Loading ---
all_vectors_list = []

print("Starting data loading process...")
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Attempting 'datasets/{file_path}'...")
        alternative_path = os.path.join('datasets', file_path)
        if not os.path.exists(alternative_path):
            raise FileNotFoundError(
                f"The file {file_path} (and {alternative_path}) does not exist. "
                "Please ensure the ARCENE dataset is correctly placed."
            )
        file_path = alternative_path

    print(f"Reading from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(tqdm(file, desc=f"Reading {os.path.basename(file_path)}")):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    all_vectors_list.append(np.array(parts, dtype=float))
                except ValueError as e:
                    print(f"Warning: Could not convert line {line_number+1} in {file_path} to float: '{line}'. Error: {e}. Skipping line.")
                    continue
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        raise

if not all_vectors_list:
    raise ValueError("No data loaded. Please check file paths and content.")

all_vectors_np = np.array(all_vectors_list)

print(f"\nSuccessfully loaded {all_vectors_np.shape[0]} vectors (samples).")
if all_vectors_np.shape[0] > 0:
    print(f"Initial dimension of each vector (number of features): {all_vectors_np.shape[1]}")
else:
    raise ValueError("Vector array is empty after processing files.")

# --- Feature Filtering for Sparsity ---
print("\n--- Filtering sparse features ---")
num_samples, num_original_features = all_vectors_np.shape
all_vectors_processed_np = all_vectors_np.copy() # Default to original if no filtering happens

if num_samples == 0 or num_original_features == 0:
    print("No samples or no features loaded, skipping feature filtering.")
else:
    zero_counts_per_column = (all_vectors_np == 0).sum(axis=0)
    zero_proportions_per_column = zero_counts_per_column / num_samples

    columns_to_keep_based_on_threshold_indices = np.where(zero_proportions_per_column <= SPARSITY_THRESHOLD)[0]
    num_features_after_threshold = len(columns_to_keep_based_on_threshold_indices)
    num_features_removed_by_threshold = num_original_features - num_features_after_threshold

    if num_features_after_threshold < MIN_FEATURES_TO_KEEP:
        print(f"Filtering by threshold {SPARSITY_THRESHOLD*100:.1f}% would leave {num_features_after_threshold} features.")
        if num_original_features <= MIN_FEATURES_TO_KEEP:
            print(f"Original number of features ({num_original_features}) is already less than or equal to MIN_FEATURES_TO_KEEP ({MIN_FEATURES_TO_KEEP}).")
            print(f"Keeping all {num_original_features} original features.")
            # all_vectors_processed_np is already a copy of all_vectors_np
        else:
            print(f"This is less than MIN_FEATURES_TO_KEEP ({MIN_FEATURES_TO_KEEP}). Selecting the top {MIN_FEATURES_TO_KEEP} least sparse features instead.")
            sorted_indices_by_sparsity = np.argsort(zero_proportions_per_column) # Ascending, so least sparse first
            columns_to_keep_indices = sorted_indices_by_sparsity[:MIN_FEATURES_TO_KEEP]
            all_vectors_processed_np = all_vectors_np[:, columns_to_keep_indices]
            num_features_actually_removed = num_original_features - MIN_FEATURES_TO_KEEP
            print(f"Retained {MIN_FEATURES_TO_KEEP} features. Removed {num_features_actually_removed} features (originally {num_features_removed_by_threshold} would have been removed by threshold).")
    else:
        if num_features_removed_by_threshold > 0:
            all_vectors_processed_np = all_vectors_np[:, columns_to_keep_based_on_threshold_indices]
            print(f"Removed {num_features_removed_by_threshold} columns (features) with more than {SPARSITY_THRESHOLD*100:.1f}% zero values.")
            print(f"Retaining {all_vectors_processed_np.shape[1]} features out of {num_original_features}.")
        else:
            print(f"No columns met removal criteria based on threshold {SPARSITY_THRESHOLD*100:.1f}%. All {num_original_features} features retained.")
            # all_vectors_processed_np is already a copy of all_vectors_np

print(f"Shape of data after feature filtering: {all_vectors_processed_np.shape}")

# --- Data Splitting ---
# Check that there are enough vectors for the requested training and testing samples
total_required_samples = num_training_samples + num_testing_samples
if all_vectors_processed_np.shape[0] < total_required_samples:
    raise ValueError(
        f"Not enough unique vectors for the combined training and testing sets. "
        f"Total required: {total_required_samples}, available: {all_vectors_processed_np.shape[0]}."
    )

# Generate a permutation of all available indices using the global seed
all_indices = np.arange(all_vectors_processed_np.shape[0])
shuffled_indices = np.random.permutation(all_indices)

# Select indices for the training set
training_indices = shuffled_indices[:num_training_samples]

# Select indices for the testing set (ensuring no overlap)
testing_indices = shuffled_indices[num_training_samples : num_training_samples + num_testing_samples]

# Retrieve the training vectors
training_vectors = all_vectors_processed_np[training_indices]

# Retrieve the testing vectors
testing_vectors = all_vectors_processed_np[testing_indices]

# --- Save Output ---
# Save training vectors
training_output_filename = f'training_vectors_{num_training_samples}_Arcene.npy' # Added _filtered
np.save(training_output_filename, training_vectors)
print(f"\nTraining vectors ({training_vectors.shape[0]} samples, {training_vectors.shape[1]} features) saved to '{training_output_filename}'.")

# Save testing vectors
testing_output_filename = f'testing_vectors_{num_testing_samples}_Arcene.npy' # Added _filtered
np.save(testing_output_filename, testing_vectors)
print(f"Testing vectors ({testing_vectors.shape[0]} samples, {testing_vectors.shape[1]} features) saved to '{testing_output_filename}'.")

# --- Verification (Optional) ---
shared_indices_check = np.intersect1d(training_indices, testing_indices)
if len(shared_indices_check) == 0:
    print("\nVerification successful: No overlap between training and testing set indices.")
else:
    print(f"\nVerification FAILED: Found {len(shared_indices_check)} overlapping indices.")

print(f"Total unique samples used: {len(training_indices) + len(testing_indices)}")
print("Data splitting, filtering, and saving complete.")