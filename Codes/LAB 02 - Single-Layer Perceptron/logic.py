# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Demonstration of the Perceptron and its performance on basic logic functions

import numpy as np

# Input data for logic functions (2 inputs per sample)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Target outputs for the AND logic function
ANDtargets = np.array([[0], [0], [0], [1]])

# Target outputs for the OR logic function
ORtargets = np.array([[0], [1], [1], [1]])

# Target outputs for the XOR logic function
XORtargets = np.array([[0], [1], [1], [0]])

# Import the perceptron (pcn) class from an external file
import pcn_logic_eg

# Demonstration of perceptron learning on the AND logic function
print("AND logic function")
pAND = pcn_logic_eg.pcn(inputs, ANDtargets)  # Initialize perceptron with inputs and AND targets
pAND.pcntrain(inputs, ANDtargets, 0.25, 6)   # Train the perceptron with learning rate 0.25 for 6 iterations

# Demonstration of perceptron learning on the OR logic function
print("OR logic function")
pOR = pcn_logic_eg.pcn(inputs, ORtargets)    # Initialize perceptron with inputs and OR targets
pOR.pcntrain(inputs, ORtargets, 0.25, 6)     # Train the perceptron with learning rate 0.25 for 6 iterations

# Demonstration of perceptron learning on the XOR logic function
print("XOR logic function")
pXOR = pcn_logic_eg.pcn(inputs, XORtargets)  # Initialize perceptron with inputs and XOR targets
pXOR.pcntrain(inputs, XORtargets, 0.25, 6)   # Train the perceptron with learning rate 0.25 for 6 iterations
