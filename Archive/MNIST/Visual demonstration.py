import struct
import numpy as np
import matplotlib.pyplot as plt

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

# Check shape
print("Shape of image vector array:", image_vectors.shape)
for i in range(5):
    print(f"Image #{i} vector (length {image_vectors[i].shape[0]}):")
    print(image_vectors[i])
    print("-" * 80)