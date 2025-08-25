# -*- coding: utf-8 -*-
"""
Created on 09-12-2024

@author: Muhammad Khuram 21013122-008
"""

import numpy as np

# Input data
inputs = np.array([0.7, 0.6, 0.9])  # [x1, x2, x3] from lecture_13_notepad

# Initial weights for 6 neurons and 3 input dimensions
weights = np.array([
    [0.31, 0.22, 0.41, 0.23, 0.71, 0.11],  # weights for input x1
    [0.21, 0.34, 0.51, 0.97, 0.63, 0.29],  # weights for input x2
    [0.39, 0.42, 0.17, 0.19, 0.55, 0.72]   # weights for input x3
])

# Hyperparameters
eta = 0.5  # Learning rate
sigma = 3  # Initial neighborhood radius
lembda = 2  # Time constant for neighborhood decay

# Neighborhood function (h): defines how the influence of the neighborhood decreases with iterations
h = lambda t: sigma * np.exp(-t / lembda)

# SOM algorithm
def som_update(inputs, weights, eta, t, sigma, lembda):
    """
    Updates the weights of the neurons based on the SOM algorithm.
    
    Parameters:
    - inputs: The input vector [x1, x2, x3].
    - weights: The weight matrix (dimensions: input_dim x num_neurons).
    - eta: Learning rate.
    - t: Current iteration.
    - sigma: Initial neighborhood radius.
    - lembda: Time constant for the decay of neighborhood influence.

    Returns:
    - Updated weights.
    - Distances between the input and neurons.
    - BMU index (index of the best-matching neuron).
    """
    num_neurons = weights.shape[1]  # Number of neurons
    h_t = h(t)  # Neighborhood influence at iteration t

    # Step 1: Calculate distances between input vector and all neurons
    distances = np.linalg.norm(inputs[:, None] - weights, axis=0)

    # Step 2: Find the index of the Best-Matching Unit (BMU) - the neuron closest to the input
    bmu_index = np.argmin(distances)

    # Step 3: Update weights of all neurons
    for j in range(num_neurons):
        if j == bmu_index:  # Update BMU's weights directly
            weights[:, j] += eta * (inputs - weights[:, j])
        else:  # Update weights of other neurons based on neighborhood function
            weights[:, j] += eta * h_t * (inputs - weights[:, j])

    return weights, distances, bmu_index

# Run the SOM for a specified number of iterations
num_iterations = 5  # Number of iterations
for t in range(num_iterations):
    print(f"\nIteration {t + 1}:")
    
    # Update weights and get distances and BMU index
    weights, distances, bmu_index = som_update(inputs, weights, eta, t, sigma, lembda)
    
    # Print distances of input from each neuron
    print("Distances:", np.round(distances, 2))
    
    # Print the index of the Best-Matching Unit (BMU)
    print("BMU Index:", bmu_index)
    
    # Print the updated weights after the current iteration
    print("Updated weights:")
    print(np.round(weights, 2))
