# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np  # Import the NumPy library for numerical operations
import mlp          # Import the MLP class from an external file

# Define data for the AND logic function
# Each row contains two inputs and the corresponding target output
anddata = np.array([
    [0, 0, 0],  # Input [0, 0] -> Output 0
    [0, 1, 0],  # Input [0, 1] -> Output 0
    [1, 0, 0],  # Input [1, 0] -> Output 0
    [1, 1, 1]   # Input [1, 1] -> Output 1
])

# Define data for the XOR logic function
# Each row contains two inputs and the corresponding target output
xordata = np.array([
    [0, 0, 0],  # Input [0, 0] -> Output 0
    [0, 1, 1],  # Input [0, 1] -> Output 1
    [1, 0, 1],  # Input [1, 0] -> Output 1
    [1, 1, 0]   # Input [1, 1] -> Output 0
])

# Train and evaluate an MLP on the AND logic function
print("Training MLP on AND logic function")
p = mlp.mlp(anddata[:, 0:2], anddata[:, 2:3], 2)  # Initialize the MLP with 2 hidden units
p.mlptrain(anddata[:, 0:2], anddata[:, 2:3], 0.25, 1001)  # Train with learning rate 0.25 for 1001 epochs
p.confmat(anddata[:, 0:2], anddata[:, 2:3])  # Print the confusion matrix and accuracy for AND data

# Train and evaluate an MLP on the XOR logic function
print("\nTraining MLP on XOR logic function")
q = mlp.mlp(xordata[:, 0:2], xordata[:, 2:3], 2, outtype='logistic')  # Use logistic activation function
q.mlptrain(xordata[:, 0:2], xordata[:, 2:3], 0.25, 5001)  # Train with learning rate 0.25 for 5001 epochs
q.confmat(xordata[:, 0:2], xordata[:, 2:3])  # Print the confusion matrix and accuracy for XOR data

#anddata = np.array([[0,0,1,0],[0,1,1,0],[1,0,1,0],[1,1,0,1]])
#xordata = np.array([[0,0,1,0],[0,1,0,1],[1,0,0,1],[1,1,1,0]])
#
#p = mlp.mlp(anddata[:,0:2],anddata[:,2:4],2,outtype='linear')
#p.mlptrain(anddata[:,0:2],anddata[:,2:4],0.25,1001)
#p.confmat(anddata[:,0:2],anddata[:,2:4])
#
#q = mlp.mlp(xordata[:,0:2],xordata[:,2:4],2,outtype='linear')
#q.mlptrain(xordata[:,0:2],xordata[:,2:4],0.15,5001)
#q.confmat(xordata[:,0:2],xordata[:,2:4])
