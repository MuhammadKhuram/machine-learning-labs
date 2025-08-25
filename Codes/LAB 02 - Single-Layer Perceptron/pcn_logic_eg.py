# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

class pcn:
    """ A basic Perceptron implementation. 
    This class includes initialization, training, forward propagation, and confusion matrix computation.
    """
    
    def __init__(self, inputs, targets):
        """ Constructor: Initializes the perceptron network with given inputs and targets.
        Args:
            inputs (numpy array): Input data for the perceptron.
            targets (numpy array): Target outputs for the input data.
        """
        # Determine the number of input features
        if np.ndim(inputs) > 1:
            self.nIn = np.shape(inputs)[1]
        else: 
            self.nIn = 1

        # Determine the number of output features
        if np.ndim(targets) > 1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        # Number of data samples
        self.nData = np.shape(inputs)[0]
        
        # Initialize weights randomly in the range [-0.05, 0.05]
        self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05

    def pcntrain(self, inputs, targets, eta, nIterations):
        """ Trains the perceptron using the perceptron learning rule.
        Args:
            inputs (numpy array): Training input data.
            targets (numpy array): Target outputs.
            eta (float): Learning rate.
            nIterations (int): Number of training iterations.
        """
        # Add bias input (-1) to each input sample
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        
        # Training loop
        for n in range(nIterations):
            # Compute activations (forward pass)
            self.activations = self.pcnfwd(inputs)
            # Update weights based on the error
            self.weights -= eta * np.dot(np.transpose(inputs), self.activations - targets)
            # Print weights and final outputs for monitoring
            print("Iteration: ", n)
            print(self.weights)
            activations = self.pcnfwd(inputs)
            print("Final outputs are:")
            print(activations)

    def pcnfwd(self, inputs):
        """ Forward pass: Computes the perceptron activations.
        Args:
            inputs (numpy array): Input data (including bias).
        Returns:
            numpy array: Binary activations (0 or 1).
        """
        # Compute activations as the dot product of inputs and weights
        activations = np.dot(inputs, self.weights)
        # Apply threshold (step function) to activations
        return np.where(activations > 0, 1, 0)

    def confmat(self, inputs, targets):
        """ Computes and prints the confusion matrix for the model predictions.
        Args:
            inputs (numpy array): Input data for testing.
            targets (numpy array): True target values for the input data.
        """
        # Add bias input (-1) to each input sample
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        # Compute raw outputs
        outputs = np.dot(inputs, self.weights)
        
        # Determine the number of classes
        nClasses = np.shape(targets)[1]

        if nClasses == 1:
            # Binary classification: Apply threshold
            nClasses = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            # Multi-class classification: Use argmax to find class predictions
            outputs = np.argmax(outputs, axis=1)
            targets = np.argmax(targets, axis=1)

        # Initialize confusion matrix
        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                # Count occurrences for each (predicted, true) pair
                cm[i, j] = np.sum(
                    np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0)
                )

        # Print confusion matrix and accuracy
        print(cm)
        print(np.trace(cm) / np.sum(cm))
