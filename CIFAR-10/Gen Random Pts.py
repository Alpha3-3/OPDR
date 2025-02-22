import pickle
import numpy as np
import random
import os

# Function to unpickle a CIFAR-10 batch file
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# Set random seeds for reproducibility
random.seed(1)
np.random.seed(1)

# Path to the CIFAR-10 batch 1 file
data_batch_path = 'cifar-10-batches-py/data_batch_1'
if not os.path.exists(data_batch_path):
    raise FileNotFoundError(f"The file {data_batch_path} does not exist")

# Load the batch file
data_dict = unpickle(data_batch_path)

# Extract the image data
# Note: In Python 3, keys are bytes. If you prefer strings, you may decode them.
images = data_dict[b'data']  # shape: (10000, 3072)

# Number of images to randomly select
num_points = 1000
if images.shape[0] < num_points:
    raise ValueError("The batch does not contain enough images.")

# Randomly select indices
selected_indices = random.sample(range(images.shape[0]), num_points)
selected_images = images[selected_indices]

# Save the selected images to a .npy file
output_file = 'testing_vectors_1000.npy'
np.save(output_file, selected_images)

print("Selected vectors have been saved successfully to", output_file)
