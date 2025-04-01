import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the digits dataset
data = load_digits()
X, y = data.data, data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define a range of target dimensions for PCA
target_dims = range(2, 21)  # from 2 to 20 dimensions
accuracies = []

# Loop over the target dimensions
for dim in target_dims:
    # Create and fit the PCA transformer
    pca = PCA(n_components=dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train a KNN classifier on the PCA-reduced data
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train_pca, y_train)

    # Evaluate accuracy on the test set and store the result
    acc = knn.score(X_test_pca, y_test)
    accuracies.append(acc)
    print(f"Target Dimension: {dim:2d} | KNN Accuracy: {acc:.4f}")

# Plot the KNN accuracy vs. PCA target dimensions
plt.figure(figsize=(8, 5))
plt.plot(target_dims, accuracies, marker='o')
plt.xlabel('PCA Target Dimension (n_components)')
plt.ylabel('KNN Accuracy')
plt.title('KNN Accuracy After PCA vs. Target Dimension')
plt.grid(True)
plt.show()
