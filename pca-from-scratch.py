"""Principal Component Analysis (PCA)"""
# Unsupervised learning technique for dimensionality reduction.
# Identify correlations and patterns in data.
# Projects data onto the directions of maximum variance (Principal Components).
# Steps:
# 1. Standardize the data (Mean = 0, Variance = 1)
# 2. Compute Covariance Matrix
# 3. Compute Eigenvalues and Eigenvectors of Covariance Matrix
# 4. Sort Eigenvalues/vectors
# 5. Transform data

import numpy as np # Used for matrix operations and eigendecomposition

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance: Function needs samples as columns
        cov = np.cov(X_centered.T)

        # Eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

if __name__ == "__main__":
    # Example Usage
    # Generate random data
    # x1 = rand, x2 = 2*x1 + rand (highly correlated)
    X = np.array([
        [1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12]
    ])
    
    print("Original Shape:", X.shape)
    
    pca = PCA(n_components=1)
    pca.fit(X)
    X_projected = pca.transform(X)
    
    print("Projected Shape:", X_projected.shape)
    print("Components (Principal Axes):", pca.components)
    print("Projected Data (1D):", X_projected)
