
import numpy as np
import pylab as pl
import svm  

# Covariance matrix for generating Gaussian-distributed data
cov = [[1.5, 1.0], [1.0, 1.5]]

# Lambda function to generate data points from two Gaussian distributions
# The two distributions are concatenated to form a single dataset
generate_data = lambda mean1, mean2: np.concatenate((
    np.random.multivariate_normal(mean1, cov, 50),  # First Gaussian cluster
    np.random.multivariate_normal(mean2, cov, 50)   # Second Gaussian cluster
))

# Generate training data for two classes (not linearly separable)
train0 = generate_data([-1, 2], [1, -1])  # Class 0: Two clusters
train1 = generate_data([4, -4], [-4, 4])  # Class 1: Two clusters

# Generate test data for the same two classes
test0 = generate_data([-1, 2], [1, -1])  # Class 0: Two clusters
test1 = generate_data([4, -4], [-4, 4])  # Class 1: Two clusters

# Combine training and test data for both classes
train = np.concatenate((train0, train1))  # Full training set
test = np.concatenate((test0, test1))    # Full test set

# Create labels for training data (+1 for Class 0, -1 for Class 1)
labeltrain = np.concatenate((np.ones((len(train0), 1)), -np.ones((len(train1), 1))))

# Create labels for test data (+1 for Class 0, -1 for Class 1)
labeltest = np.concatenate((np.ones((len(test0), 1)), -np.ones((len(test1), 1))))

# Visualization of training data
pl.figure()
pl.plot(train0[:, 0], train0[:, 1], "o", color=(0.9,0,0.8))  # Plot Class 0
pl.plot(train1[:, 0], train1[:, 1], "o", color=(0.0, 0.8, 0.7))  # Plot Class 1

# Initialize and train the SVM with a linear kernel
svm = svm.svm(kernel='linear', C=0.1)
svm.train_svm(train, labeltrain)

# Highlight support vectors used by the SVM
pl.scatter(svm.X[:, 0], svm.X[:, 1], s=200, color='k')  # Mark support vectors

# Test the SVM model on the test data
predict = svm.classifier(test, soft=False)  # Predict class labels for test data
accuracy = np.mean(predict == labeltest) * 100  # Calculate accuracy
print(accuracy, "% test accuracy")  # Display test accuracy

# Visualize decision boundary and margins
x, y = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))  # Create a grid
points = np.c_[x.ravel(), y.ravel()]  # Flatten the grid points for classification
outpoints = svm.classifier(points, soft=True).reshape(x.shape)  # Get SVM output

# Draw decision boundary (level 0) and margins (levels +1 and -1)
pl.contour(x, y, outpoints, levels=[0.0], colors='k', linewidths=3)  # Decision boundary
pl.contour(x, y, outpoints + 1, levels=[0.0], colors='k', linewidths=2)  # Margin +1
pl.contour(x, y, outpoints - 1, levels=[0.0], colors='k', linewidths=2)  # Margin -1

# Adjust plot aesthetics
pl.axis("tight")
pl.show()
