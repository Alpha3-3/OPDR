import os
import numpy as np
from PIL import Image

# Directory containing the .jpg files
directory = 'val2017'
# Get list of all .jpg files in the directory
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.jpg')]

# Check if there are at least 600 files
if len(files) < 600:
    raise ValueError("Not enough .jpg files found in the directory.")

# Set random seed for reproducibility and randomly select 600 files
np.random.seed(1)
selected_files = np.random.choice(files, 600, replace=False)

vectors = []
for file in selected_files:
    # Open image and convert to RGB to ensure consistency
    img = Image.open(file).convert('RGB')
    # Resize image to 256x256
    img_resized = img.resize((32, 32))
    # Convert image to numpy array and flatten it
    img_vector = np.array(img_resized).flatten()
    vectors.append(img_vector)

# Convert list of vectors to a NumPy array. Each vector will have length 256*256*3 = 196608.
vectors = np.array(vectors)

# Save the resulting vectors to a .npy file
np.save('training_vectors_600_Coco.npy', vectors)

np.random.seed(2)
selected_files = np.random.choice(files, 600, replace=False)

vectors = []
for file in selected_files:
    # Open image and convert to RGB to ensure consistency
    img = Image.open(file).convert('RGB')
    # Resize image to 256x256
    img_resized = img.resize((32, 32))
    # Convert image to numpy array and flatten it
    img_vector = np.array(img_resized).flatten()
    vectors.append(img_vector)

# Convert list of vectors to a NumPy array. Each vector will have length 256*256*3 = 196608.
vectors = np.array(vectors)

# Save the resulting vectors to a .npy file
np.save('testing_vectors_600_Coco.npy', vectors)