# -*- coding: utf-8 -*-
"""
Created on 27-11-2024

@author: Muhammad Khuram 21013122-008
"""

import numpy as np
import matplotlib.pyplot as plt

# PCA function that reduces the dimensionality of the data
def pca(data, nRedDim=0, normalise=1):
    """
    Perform Principal Component Analysis (PCA) to reduce the dimensionality of the data.
    
    Parameters:
    data (ndarray): The input data (rows as samples, columns as features)
    nRedDim (int): The number of dimensions to reduce the data to (0 for no reduction)
    normalise (int): Whether to normalize the eigenvectors (1 for normalize, 0 for no normalization)
    
    Returns:
    x (ndarray): The data projected onto the principal components (reduced data)
    y (ndarray): The reconstructed data from the reduced components
    evals (ndarray): Eigenvalues corresponding to the principal components
    evecs (ndarray): Eigenvectors corresponding to the principal components
    """
    # Centre data by subtracting the mean of each feature (mean normalization)
    m = np.mean(data, axis=0)
    data -= m

    # Calculate the covariance matrix of the data
    C = np.cov(np.transpose(data))

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    evals, evecs = np.linalg.eig(C)
    
    # Sort eigenvalues in descending order and reorder eigenvectors accordingly
    indices = np.argsort(evals)
    indices = indices[::-1]  # Reverse to get descending order
    evecs = evecs[:, indices]
    evals = evals[indices]

    # If a specific number of reduced dimensions is requested, truncate the eigenvectors
    if nRedDim > 0:
        evecs = evecs[:, :nRedDim]
    
    # Normalize the eigenvectors (optional step)
    if normalise:
        for i in range(np.shape(evecs)[1]):
            evecs[:, i] /= np.linalg.norm(evecs[:, i]) * np.sqrt(evals[i])

    # Project the data onto the principal components to get the reduced data
    x = np.dot(np.transpose(evecs), np.transpose(data))
    
    # Reconstruct the original data from the reduced data
    y = np.transpose(np.dot(evecs, x)) + m
    
    return x, y, evals, evecs

# Example data: A simple 2D dataset with 10 samples
data = np.array([[2.5, 2.4],  # Sample 1
                 [0.5, 0.7],  # Sample 2
                 [2.2, 2.9],  # Sample 3
                 [1.9, 2.2],  # Sample 4
                 [3.1, 3.0],  # Sample 5
                 [2.3, 2.7],  # Sample 6
                 [2.0, 1.6],  # Sample 7
                 [1.0, 1.1],  # Sample 8
                 [1.5, 1.6],  # Sample 9
                 [1.1, 0.9]]) # Sample 10

# Call the PCA function to reduce the data to 1 dimension
nRedDim = 1  # Reduce to 1 dimension
x, y, evals, evecs = pca(data, nRedDim=nRedDim, normalise=1)

# Print the results: reduced data, reconstructed data, eigenvalues, and eigenvectors
print("Reduced Data (x):")
print(x)
print("\nReconstructed Data (y):")
print(y)
print("\nEigenvalues (evals):")
print(evals)
print("\nEigenvectors (evecs):")
print(evecs)

# Plotting the original and reduced data along with the principal components

# Create a new figure for the plot with specified size
plt.figure(figsize=(8, 6))

# Plot the original data points
plt.scatter(data[:, 0], data[:, 1], color='blue', marker='*', s=200, label='Original Data')

# Plot the reduced data points (after PCA) along the x-axis (1D)
plt.scatter(x[0, :], np.zeros_like(x[0, :]), color='red', s=100, label='Reduced Data')

# Plot the principal components (eigenvectors) as arrows
for i in range(evecs.shape[1]):
    plt.quiver(0, 0, evecs[0, i], evecs[1, i], angles='xy', scale_units='xy', scale=1, color='green')

# Set axis labels and the plot title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA: Original and Reduced Data')

# Add a legend to the plot
plt.legend()

# Enable grid for better visualization of the points
plt.grid(True)

# Display the plot
plt.show()
