import numpy as np
import pcn_logic_eg as pcn
                                     
class rbf:
    """ Radial Basis Function network
    Parameters are number of RBFs, their width (sigma), how to train the network 
    (pseudo-inverse or random selection) and whether the RBFs are normalised
    """

    def __init__(self, inputs, targets, nRBF, sigma=0, normalise=0):
        # Initialize network parameters
        self.nin = np.shape(inputs)[1]  # Number of input features
        self.nout = np.shape(targets)[1]  # Number of output classes
        self.ndata = np.shape(inputs)[0]  # Number of training data points
        self.nRBF = nRBF  # Number of RBFs (hidden neurons)
        self.normalise = normalise  # Flag for normalization
        
        # Initialize hidden layer (input + bias)
        self.hidden = np.zeros((self.ndata, self.nRBF + 1))  # Include bias as last column
        
        # Determine the width (sigma) of the Gaussians if not provided
        if sigma == 0:
            # Calculate sigma based on the range of the input data
            d = (inputs.max(axis=0) - inputs.min(axis=0)).max()
            self.sigma = d / np.sqrt(2 * nRBF)  # Default sigma calculation
        else:
            self.sigma = sigma  # Use provided sigma value

        # Initialize the perceptron (for training output weights)
        self.perceptron = pcn.pcn(self.hidden[:, :-1], targets)  # Exclude bias from hidden layer
        
        # Initialize RBF centers (weights1) to zero initially
        self.weights1 = np.zeros((self.nin, self.nRBF))
        
    def rbftrain(self, inputs, targets, eta=0.25, niterations=100):
        """ Train the RBF network with the given inputs and targets """
        
        # Select random points from the dataset to serve as RBF centers
        indices = list(range(self.ndata))
        np.random.shuffle(indices)
        for i in range(self.nRBF):
            self.weights1[:, i] = inputs[indices[i], :]  # Assign random input points to RBF centers

        # Compute activations for the hidden layer based on RBFs (Gaussian functions)
        for i in range(self.nRBF):
            # Calculate the Gaussian activation for each RBF center
            self.hidden[:, i] = np.exp(-np.sum((inputs - np.ones((1, self.nin)) * self.weights1[:, i]) ** 2, axis=1) / (2 * self.sigma ** 2))

        # Normalize the hidden layer activations if specified
        if self.normalise:
            self.hidden[:, :-1] /= np.transpose(np.ones((1, np.shape(self.hidden)[0])) * self.hidden[:, :-1].sum(axis=1))

        # Train the perceptron (output layer) using the computed activations
        self.perceptron.pcntrain(self.hidden[:, :-1], targets, eta, niterations)
        
    def rbffwd(self, inputs):
        """ Forward pass through the RBF network to get predictions """
        
        # Initialize hidden layer activations for the given inputs
        hidden = np.zeros((np.shape(inputs)[0], self.nRBF + 1))  # Include bias
        
        # Compute activations for the hidden layer (RBFs)
        for i in range(self.nRBF):
            hidden[:, i] = np.exp(-np.sum((inputs - np.ones((1, self.nin)) * self.weights1[:, i]) ** 2, axis=1) / (2 * self.sigma ** 2))

        # Normalize the hidden layer activations if specified
        if self.normalise:
            hidden[:, :-1] /= np.transpose(np.ones((1, np.shape(hidden)[0])) * hidden[:, :-1].sum(axis=1))
        
        # Add the bias term (last column)
        hidden[:, -1] = -1

        # Pass activations to perceptron and get outputs
        outputs = self.perceptron.pcnfwd(hidden)
        return outputs
    
    def confmat(self, inputs, targets):
        """ Generate a confusion matrix for model evaluation """
        
        # Get network's predictions for the input data
        outputs = self.rbffwd(inputs)
        nClasses = np.shape(targets)[1]  # Number of classes in the target

        # If it's a binary classification, convert output to 0 or 1
        if nClasses == 1:
            nClasses = 2  # Treat as binary classification
            outputs = np.where(outputs > 0, 1, 0)
        else:
            # For multi-class, apply 1-of-N encoding and get the predicted class
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        # Initialize confusion matrix
        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                # Count occurrences of each pair of predicted and actual classes
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        # Print the confusion matrix
        print(cm)
        # Print the accuracy (trace of the matrix / total sum)
        print(np.trace(cm) / np.sum(cm))
