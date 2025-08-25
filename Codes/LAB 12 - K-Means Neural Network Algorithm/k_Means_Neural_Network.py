# -*- coding: utf-8 -*-
"""
Created on 30-11-2024

@author: Muhammad Khuram 21013122-008
"""

import numpy as np

class kmeans:
    """The k-Means Algorithm implemented as a neural network"""

    def __init__(self, k, data, nEpochs, eta):
        """
        Initializes the k-Means model.
        :param k: Number of clusters
        :param data: Input data points (numpy array)
        :param nEpochs: Number of training epochs
        :param eta: Learning rate for centroid updates
        """
        self.nData = np.shape(data)[0]  # Number of data points
        self.nDim = np.shape(data)[1]  # Dimensionality of each data point
        self.k = k  # Number of clusters
        self.nEpochs = nEpochs  # Number of training iterations
        # Randomly initialize centroids by selecting k points from the data
        self.weights = data[np.random.choice(data.shape[0], k, replace=False), :]
        self.eta = eta  # Learning rate

    def kmeanstrain(self, data):
        """
        Trains the k-Means model using the competitive learning rule.
        :param data: Input data points (numpy array)
        """
        for i in range(self.nEpochs):  # Loop over all epochs
            for j in range(self.nData):  # Loop over all data points
                # Compute activation values for each centroid (similarity measure)
                activation = np.sum(self.weights * np.transpose(data[j:j+1, :]), axis=1)
                # Identify the winning centroid (closest to the data point)
                winner = np.argmax(activation)
                # Update the winning centroid using the learning rule
                self.weights[winner, :] += self.eta * (data[j, :] - self.weights[winner, :])

    def kmeansfwd(self, data):
        """
        Assigns each data point to the nearest centroid.
        :param data: Input data points (numpy array)
        :return: Array of cluster labels for each data point
        """
        best = np.zeros(np.shape(data)[0])  # Initialize array to store cluster labels
        for i in range(np.shape(data)[0]):  # Loop over all data points
            # Compute activation values for each centroid
            activation = np.sum(self.weights * np.transpose(data[i:i+1, :]), axis=1)
            # Assign the data point to the nearest centroid
            best[i] = np.argmax(activation)
        return best

    def get_centroids(self):
        """
        Returns the final centroids after training.
        :return: Centroids as a numpy array
        """
        return self.weights

# Sample data points from example in lecture_10_notepad
data = np.array([[1, 1], [1.5, 2.0], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]])
k = 2  # Number of clusters to create

# Instantiate and train the k-Means model
model = kmeans(k, data, 5000, 0.1)  # 5000 epochs with a learning rate of 0.1
model.kmeanstrain(data)  # Train the model with the data

# Get the final centroids and cluster labels
centroids = model.get_centroids()  # Retrieve the final centroid positions
labels = model.kmeansfwd(data)  # Assign each data point to the nearest centroid

# Output the centroids and the corresponding cluster labels
print("Centroids:")
print(np.round(centroids,2))
print("Labels:")
print(labels)


