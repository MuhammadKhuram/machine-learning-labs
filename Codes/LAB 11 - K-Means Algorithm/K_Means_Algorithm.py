# -*- coding: utf-8 -*-
"""
Created on 30-11-2024

@author: Muhammad Khuram 21013122-008
"""

import numpy as np

def k_means(data, k, max_iterations=100):
    # Randomly initialize the centroids by selecting k points from the dataset
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    # Iterate up to the maximum number of iterations
    for _ in range(max_iterations):
        # Compute the distance between each point and the centroids, then assign the point to the nearest centroid
        labels = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
        
        # Update the centroids by calculating the mean of all points assigned to each centroid
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # If centroids do not change, the algorithm has converged, so break the loop
        if np.all(centroids == new_centroids):
            break

        # Update centroids for the next iteration
        centroids = new_centroids

    # Return the final centroids and the cluster labels for each point
    return centroids, labels

# data (2D points) from example in lecture_10_notepad
data = np.array([[1, 1], [1.5, 2.0], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]])
k = 2  # Number of clusters

# Call the k_means function to cluster the data
centroids, labels = k_means(data, k)

# Output the final centroids and the corresponding labels for each data point
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
