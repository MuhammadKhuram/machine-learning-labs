# -*- coding: utf-8 -*-
"""
Created on 27-11-2024

@author: Muhammad Khuram 21013122-008
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la

def lda(data, labels, redDim):
    """
    Perform Linear Discriminant Analysis (LDA) for dimensionality reduction.

    Parameters:
        data: numpy.ndarray
            Input data matrix (samples x features).
        labels: numpy.ndarray
            Class labels for each data point.
        redDim: int
            Number of dimensions to reduce the data to.

    Returns:
        newData: numpy.ndarray
            Data projected into the reduced dimensional space.
        w: numpy.ndarray
            Projection matrix (features x redDim).
    """
    # Center the data by subtracting the mean
    data -= data.mean(axis=0)
    print(data);
    
    # Initialize scatter matrices
    Sw = np.zeros((data.shape[1], data.shape[1]))  # Within-class scatter matrix
    Sb = np.zeros((data.shape[1], data.shape[1]))  # Between-class scatter matrix
    
    # Calculate scatter matrices for each class
    classes = np.unique(labels)
    for cls in classes:
        # Extract data points belonging to the current class
        class_data = data[labels == cls]
        # Compute class mean
        class_mean = class_data.mean(axis=0)
        # Compute within-class scatter
        Sw += np.dot((class_data - class_mean).T, (class_data - class_mean))
        # Compute between-class scatter
        mean_diff = (class_mean).reshape(-1, 1)
        Sb += len(class_data) * np.dot(mean_diff, mean_diff.T)
    
    # Solve the generalized eigenvalue problem for Sw^-1 Sb
    evals, evecs = la.eig(np.linalg.pinv(Sw).dot(Sb))
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(evals)[::-1]
    w = evecs[:, sorted_indices[:redDim]]
    
    # Project the data to the new lower-dimensional space
    newData = np.dot(data, w)
    return newData, w

# Example data
data = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.35, 0.3], [0.4, 0.4], 
                 [0.6, 0.4], [0.7, 0.45], [0.75, 0.4], [0.8, 0.35]])
labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

# Perform LDA to reduce data to 1 dimension
newData, w = lda(data, labels, redDim=1)

# Plot the original data in 2D
plt.figure(figsize=(10, 6))
for cls in np.unique(labels):
    plt.scatter(data[labels == cls][:, 0], data[labels == cls][:, 1], label=f"Class {cls}")
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# Plot the reduced data in 1D
plt.figure(figsize=(8, 4))
for cls in np.unique(labels):
    plt.scatter(newData[labels == cls], [0] * len(newData[labels == cls]), label=f"Class {cls}")
plt.title("Reduced Data (1D Projection)")
plt.xlabel("LDA Feature 1")
plt.yticks([])  # Hide y-axis ticks for 1D projection
plt.legend()
plt.grid(True)
plt.show()
