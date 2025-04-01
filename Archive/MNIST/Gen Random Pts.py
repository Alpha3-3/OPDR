import struct
import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read metadata
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic number %d in MNIST image file!" % magic)

        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape((num_images, rows, cols))
    return images

# Path to your file (update this if needed)
file_path = 'MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'

# Load and inspect
images = load_mnist_images(file_path)

# Show the first 5 images
# Flatten the images into vectors (each image is a row)
image_vectors = images.reshape(images.shape[0], -1)  # shape: (10000, 784)

np.random.seed(1)
# Randomly choose 600 unique indices
num_samples = 600
random_indices = np.random.choice(image_vectors.shape[0], size=num_samples, replace=False)

# Select the vectors
selected_vectors = image_vectors[random_indices]

# Save to .npy file
np.save('training_vectors_600_MNIST.npy', selected_vectors)

print("Saved 600 random vectors to mnist_600_vectors.npy")

np.random.seed(2)
# Randomly choose 600 unique indices
num_samples = 600
random_indices = np.random.choice(image_vectors.shape[0], size=num_samples, replace=False)

# Select the vectors
selected_vectors = image_vectors[random_indices]

# Save to .npy file
np.save('testing_vectors_600_MNIST.npy', selected_vectors)

print("Saved 600 random vectors to mnist_600_vectors.npy")