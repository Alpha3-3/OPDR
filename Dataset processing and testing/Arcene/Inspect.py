import numpy as np
import pandas as pd

# Load pre-trained vectors
def load_vectors(file_path):
    return np.load(file_path)

training_vectors = load_vectors('training_vectors_600_Arcene.npy')
testing_vectors = load_vectors('testing_vectors_300_Arcene.npy')

# For inspection, only use the first 3000 dimensions
training_inspect = training_vectors
testing_inspect = testing_vectors

# Print statistics of the vectors (first 3000 dimensions)
print("Training Vectors Description (first 3000 dimensions):")
print(pd.DataFrame(training_inspect).describe())
print("\nFirst 5 rows of Training Vectors (first 3000 dimensions):")
print(pd.DataFrame(training_inspect).head().to_string())

print("\nTesting Vectors Description (first 3000 dimensions):")
print(pd.DataFrame(testing_inspect).describe())
print("\nFirst 5 rows of Testing Vectors (first 3000 dimensions):")
print(pd.DataFrame(testing_inspect).head().to_string())

# Count how many features (dimensions) in the first 3000 columns are not fully zero.
nonzero_training_features = np.sum(np.any(training_inspect != 0, axis=0))
nonzero_testing_features = np.sum(np.any(testing_inspect != 0, axis=0))

print("\nNonzero feature counts (first 3000 dimensions):")
print(f"Training vectors: {nonzero_training_features} out of {training_inspect.shape[1]} dimensions are not fully zero.")
fully_zero_training = training_inspect.shape[1] - nonzero_training_features
print(f"This means {fully_zero_training} dimensions are entirely zero in training vectors (first 3000 dims).")

print(f"Testing vectors: {nonzero_testing_features} out of {testing_inspect.shape[1]} dimensions are not fully zero.")
fully_zero_testing = testing_inspect.shape[1] - nonzero_testing_features
print(f"This means {fully_zero_testing} dimensions are entirely zero in testing vectors (first 3000 dims).")


# --- Inspection for Almost Entirely Empty Columns (Added Section) ---
print("\n--- Inspecting for Almost Entirely Empty Columns (first 3000 dimensions) ---")

def inspect_almost_empty_columns(data, data_name, threshold=0.95):
    """
    Inspects data for columns that are 'almost empty', defined as having
    a proportion of zero values greater than the specified threshold.
    """
    num_samples, num_features = data.shape
    if num_samples == 0:
        print(f"{data_name}: No data (0 samples) to inspect.")
        return

    # Calculate the proportion of zeros in each column
    # (data == 0) creates a boolean array, .sum(axis=0) counts True values (zeros) per column
    zero_counts_per_column = (data == 0).sum(axis=0)
    zero_proportions_per_column = zero_counts_per_column / num_samples

    # Identify columns where the proportion of zeros is greater than the threshold
    almost_empty_columns_indices = np.where(zero_proportions_per_column > threshold)[0]
    num_almost_empty_columns = len(almost_empty_columns_indices)

    print(f"\nAnalysis for: {data_name} (threshold for 'almost empty' > {threshold*100}% zeros per column)")
    print(f"Number of columns (features) considered 'almost entirely empty': {num_almost_empty_columns} out of {num_features}")

    if num_almost_empty_columns > 0:
        print(f"Indices of these 'almost entirely empty' columns: {almost_empty_columns_indices.tolist()}")
        # You could add more details here, e.g., the exact zero proportion for these columns
        # for idx in almost_empty_columns_indices[:min(5, num_almost_empty_columns)]: # Example for top 5
        #     print(f"  Column {idx}: {zero_proportions_per_column[idx]*100:.2f}% zeros")
    else:
        print(f"No columns found with more than {threshold*100}% zero values.")

# Perform the inspection with a 95% threshold
inspect_almost_empty_columns(training_inspect, "Training vectors (first 3000 dims)", threshold=0.95)
inspect_almost_empty_columns(testing_inspect, "Testing vectors (first 3000 dims)", threshold=0.95)

# Perform the inspection with a 99% threshold for stricter checking
inspect_almost_empty_columns(training_inspect, "Training vectors (first 3000 dims)", threshold=0.99)
inspect_almost_empty_columns(testing_inspect, "Testing vectors (first 3000 dims)", threshold=0.99)
# --- End of Added Section ---


# Updated Overlap Check:
# Round the full vectors to avoid precision issues when comparing
print("\n--- Overlap Check (Full Vectors) ---")
rounded_training_vectors = np.round(training_vectors, decimals=5)
rounded_testing_vectors = np.round(testing_vectors, decimals=5)

# Convert each vector (row) into a tuple and form sets for comparison
training_set = set(map(tuple, rounded_training_vectors))
testing_set = set(map(tuple, rounded_testing_vectors))

# Find the overlapping vectors using set intersection
overlap = training_set.intersection(testing_set)
print(f"\nNumber of overlapping vectors between full training and testing sets: {len(overlap)}")
if len(overlap) > 0:
    print("Overlap found between training and testing vectors.")
else:
    print("No overlap found between training and testing vectors.")