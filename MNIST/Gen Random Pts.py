import os
import struct
import random
import numpy as np
from array import array
from tqdm import tqdm

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        # Read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    f"Magic number mismatch in labels file, expected 2049, got {magic}"
                )
            labels = array("B", file.read())

        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size2, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    f"Magic number mismatch in images file, expected 2051, got {magic}"
                )
            if size != size2:
                raise ValueError("Mismatch in number of labels vs. images")

            image_data = array("B", file.read())

        # Convert images into a list of 2D numpy arrays
        images = []
        for i in range(size):
            start_index = i * rows * cols
            end_index = (i + 1) * rows * cols
            img = np.array(image_data[start_index:end_index])
            img = img.reshape(rows, cols)
            images.append(img)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)

#
# Main Script
#

# 1. Point this to the exact folder where your MNIST files reside.
#    For example, if your files are located at:
#    D:\My notes\UW\HPDIC Lab\OPDR\PCA vs DW_PMAD\PCA vs DW_PMAD\MNIST\MNIST
#    and inside this folder you have:
#        train-images-idx3-ubyte
#        train-labels-idx1-ubyte
#        t10k-images-idx3-ubyte
#        t10k-labels-idx1-ubyte
#
#    Then do:
mnist_dir = r"D:\My notes\UW\HPDIC Lab\OPDR\PCA vs DW_PMAD\PCA vs DW_PMAD\MNIST\MNIST"

training_images_filepath = os.path.join(mnist_dir, "train-images-idx3-ubyte")
training_labels_filepath = os.path.join(mnist_dir, "train-labels-idx1-ubyte")
test_images_filepath     = os.path.join(mnist_dir, "t10k-images-idx3-ubyte")
test_labels_filepath     = os.path.join(mnist_dir, "t10k-labels-idx1-ubyte")

# 2. Initialize the data loader and load the data
mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# 3. Flatten each 28x28 image into a 784-dimensional vector
x_train_flat = [img.flatten() for img in x_train]  # List of arrays
x_test_flat  = [img.flatten() for img in x_test]   # List of arrays

# Combine if you want to sample from the entire MNIST dataset
all_images = np.array(x_train_flat + x_test_flat)
all_labels = np.array(y_train + y_test)  # optional, if you need labels

# 4. Randomly select num_points images
num_points = 1000

# Ensure we have enough images
if len(all_images) < num_points:
    raise ValueError("Not enough images to sample from.")

# Set random seed for reproducibility
random.seed(1)
np.random.seed(1)

# Sample indices
selected_indices = random.sample(range(len(all_images)), num_points)
selected_vectors = all_images[selected_indices]

# (Optional) selected_labels = all_labels[selected_indices]  # if you want the labels

# 5. Save the selected vectors to a .npy file
npy_file_path = 'mnist_vectors_1000.npy'
np.save(npy_file_path, selected_vectors)

print(f"{num_points} MNIST vectors have been selected and saved to '{npy_file_path}'.")
