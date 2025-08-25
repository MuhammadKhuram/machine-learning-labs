# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The sinewave regression example

# Import necessary libraries
import pylab as pl
import numpy as np

# Set up the data
# Generate 40 equally spaced points between 0 and 1
x = np.linspace(0, 1, 40).reshape((40, 1))

# Generate target values (t) as a combination of sine and cosine waves with noise
t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(40).reshape((40, 1)) * 0.2

# Normalize x values to be between -1 and 1
x = (x - 0.5) * 2

# Split data into training, testing, and validation sets
# Training data: every second sample
train = x[0::2, :]
traintarget = t[0::2, :]

# Test data: every fourth sample starting from the second
test = x[1::4, :]
testtarget = t[1::4, :]

# Validation data: every fourth sample starting from the fourth
valid = x[3::4, :]
validtarget = t[3::4, :]

# Plot the generated data
pl.plot(x, t, 'o')  # Scatter plot of input vs. target
pl.xlabel('x')      # Label for x-axis
pl.ylabel('t')      # Label for y-axis

# Import the MLP (Multi-Layer Perceptron) class
import mlp

# Initialize an MLP with 3 hidden nodes and linear output type
net = mlp.mlp(train, traintarget, 3, outtype='linear')

# Train the MLP using the training data for 101 epochs with a learning rate of 0.25
net.mlptrain(train, traintarget, 0.25, 101)

# Use early stopping to avoid overfitting, validating against the validation dataset
net.earlystopping(train, traintarget, valid, validtarget, 0.25)

# Below is commented-out code to experiment with different network sizes and measure performance
# Uncomment and modify if needed for further exploration

# Test out different sizes of network
# count = 0
# out = np.zeros((10, 7))
# for nnodes in [1, 2, 3, 5, 10, 25, 50]:
#     for i in range(10):
#         net = mlp.mlp(train, traintarget, nnodes, outtype='linear')
#         out[i, count] = net.earlystopping(train, traintarget, valid, validtarget, 0.25)
#     count += 1
# 
# test = np.concatenate((test, -np.ones((np.shape(test)[0], 1))), axis=1)
# outputs = net.mlpfwd(test)
# print(0.5 * np.sum((outputs - testtarget) ** 2))
# 
# print(out)
# print(out.mean(axis=0))
# print(out.var(axis=0))
# print(out.max(axis=0))
# print(out.min(axis=0))

# Show the plotted data
pl.show()
