# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:53:38 2024

@author: THE MAD TITAN
"""

import numpy as np
import rbf as rbf

# Define XOR dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Number of RBFs
nRBF = 4  # You can adjust this based on the complexity

# Initialize RBF network
xor = rbf.rbf(inputs, targets, nRBF, sigma=0, normalise=0)

# Train the RBF network
xor.rbftrain(inputs, targets, eta=0.25, niterations=10)

# Test the RBF network
outputs = xor.rbffwd(inputs)

# Evaluate performance using confusion matrix
print("Confusion Matrix:")
xor.confmat(inputs, targets)

# Print predictions
print("Predicted Outputs:")
print(outputs)

# Print the final RBF centers (weights1)
print("Final RBF Centers (Weights1):")
print(xor.weights1)  # Accessing final RBF centers (weights1)





