# -*- coding: utf-8 -*-
"""
Created on 07-01-2025

@author: %(Muhammad Khuram 21013122-008)s
"""
import numpy as np

# Step 1: Define the signals and mixing matrix from lecture_notepad
s1 = np.array([1, 2, 3, 4])
s2 = np.array([4, 3, 2, 1])
S = np.vstack((s1, s2))

A = np.array([[2, 1], [1, 3]])  # Mixing matrix

# Step 2: Generate mixed signals
X = A @ S  # Mixed signals

# Step 3: Center and whiten the mixed signals
X_centered = X - X.mean(axis=1, keepdims=True)
cov = (X_centered @ X_centered.T) /len(s1) # Or s2

epsilon = 1e-6  # Small regularization constant
D, E = np.linalg.eigh(cov)
D = np.maximum(D, epsilon)  # Replace small or zero eigenvalues with epsilon
whitening_matrix = np.diag(1.0 / np.sqrt(D)) @ E.T
X_whitened = whitening_matrix @ X_centered


# Step 4: Perform ICA using gradient ascent
def g(x):
    return np.tanh(x)  # Non-linear function


np.random.seed(0)
W = np.random.rand(2, 2)  # Initialize unmixing matrix

for _ in range(1000):
    WX = W @ X_whitened
    grad = (X_whitened @ g(WX).T) / X_whitened.shape[1] - np.eye(W.shape[0])
    W += 0.01 * grad
    W = np.linalg.qr(W)[0]  # Ensure orthogonality

# Step 5: Recover the original signals
S_recovered = W @ X_whitened

# Print the results
print("Original Signals:")
print(S)
print("\nMixed Signals:")
print(X)
print("\nRecovered Signals:")
print(S_recovered)

