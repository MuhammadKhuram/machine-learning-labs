# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

class mlp:
    """ A Multi-Layer Perceptron (MLP) """

    def __init__(self, inputs, targets, nhidden, beta=1, momentum=0.9, outtype='logistic'):
        """ Constructor to initialize the MLP model """
        # Define the dimensions of the network
        self.nin = np.shape(inputs)[1]  # Number of input neurons
        self.nout = np.shape(targets)[1]  # Number of output neurons
        self.ndata = np.shape(inputs)[0]  # Number of data points
        self.nhidden = nhidden  # Number of hidden neurons

        self.beta = beta  # Scaling parameter for logistic/softmax activation
        self.momentum = momentum  # Momentum term for weight updates
        self.outtype = outtype  # Output neuron type ('linear', 'logistic', or 'softmax')

        # Initialize weights with small random values
        self.weights1 = (np.random.rand(self.nin + 1, self.nhidden) - 0.5) * 2 / np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden + 1, self.nout) - 0.5) * 2 / np.sqrt(self.nhidden)

    def earlystopping(self, inputs, targets, valid, validtargets, eta, niterations=100):
        """ Implements early stopping using a validation set """
        # Add bias to validation inputs
        valid = np.concatenate((valid, -np.ones((np.shape(valid)[0], 1))), axis=1)

        # Initialize validation error tracking
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        count = 0

        # Continue training until validation error converges
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001)):
            count += 1
            print(count)
            self.mlptrain(inputs, targets, eta, niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error

            # Compute validation error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5 * np.sum((validtargets - validout) ** 2)

        print("Stopped", new_val_error, old_val_error1, old_val_error2)
        return new_val_error

    def mlptrain(self, inputs, targets, eta, niterations):
        """ Trains the MLP using backpropagation """
        # Add bias to input data
        inputs = np.concatenate((inputs, -np.ones((self.ndata, 1))), axis=1)

        # Initialize weight updates
        updatew1 = np.zeros(np.shape(self.weights1))
        updatew2 = np.zeros(np.shape(self.weights2))

        for n in range(niterations):
            # Forward pass to compute outputs
            self.outputs = self.mlpfwd(inputs)

            # Compute error using mean squared error
            error = 0.5 * np.sum((self.outputs - targets) ** 2)
            if n % 100 == 0:  # Print error every 100 iterations
                print(f"Iteration: {n}, Error: {error}")

            # Compute output delta based on output type
            if self.outtype == 'linear':
                deltao = (self.outputs - targets) / self.ndata
            elif self.outtype == 'logistic':
                deltao = self.beta * (self.outputs - targets) * self.outputs * (1.0 - self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs - targets) * (self.outputs * (-self.outputs) + self.outputs) / self.ndata
            else:
                raise ValueError("Invalid output type")

            # Backpropagate errors to hidden layer
            deltah = self.hidden * self.beta * (1.0 - self.hidden) * np.dot(deltao, self.weights2.T)

            # Update weights using gradient descent with momentum
            updatew1 = eta * np.dot(inputs.T, deltah[:, :-1]) + self.momentum * updatew1
            updatew2 = eta * np.dot(self.hidden.T, deltao) + self.momentum * updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2

    def mlpfwd(self, inputs):
        """ Runs the network forward to compute outputs """
        # Compute activations for the hidden layer
        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))  # Logistic activation
        self.hidden = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)  # Add bias

        # Compute activations for the output layer
        outputs = np.dot(self.hidden, self.weights2)

        # Apply activation function based on output type
        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1) * np.ones((1, np.shape(outputs)[0]))
            return (np.exp(outputs).T / normalisers).T
        else:
            raise ValueError("Invalid output type")

    def confmat(self, inputs, targets):
        """ Computes and prints the confusion matrix """
        # Add bias to inputs
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        # Compute network outputs
        outputs = self.mlpfwd(inputs)
        nclasses = np.shape(targets)[1]

        # Threshold outputs for binary classification
        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs > 0.5, 1, 0)
        else:
            # Decode 1-of-N targets
            outputs = np.argmax(outputs, axis=1)
            targets = np.argmax(targets, axis=1)

        # Create and populate confusion matrix
        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum((outputs == i) * (targets == j))

        # Print confusion matrix and accuracy
        print("Confusion matrix is:")
        print(cm)
        print(f"Percentage Correct: {np.trace(cm) / np.sum(cm) * 100}")
