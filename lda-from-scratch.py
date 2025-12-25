"""Linear Discriminant Analysis (LDA)"""
# Supervised learning for dimensionality reduction and classification.
# Maximizes the separation between classes.
# Steps:
# 1. Compute Within-class scatter matrix (Sw)
# 2. Compute Between-class scatter matrix (Sb)
# 3. Compute eigenvalues/vectors of (Sw^-1 * Sb)
# 4. Sort and select top eigenvectors
# 5. Project data

import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # 1. S_W, S_B
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            
            # S_W: Sum of (X_c - mean_c).T * (X_c - mean_c)
            # (4, n_c) * (n_c, 4) = (4,4)
            S_W += (X_c - mean_c).T.dot((X_c - mean_c))

            # S_B: n_c * (mean_c - mean_overall).T * (mean_c - mean_overall)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1) # make column vector
            S_B += n_c * (mean_diff).dot(mean_diff.T)

        # 2. Determine SW^-1 * SB
        A = np.linalg.inv(S_W).dot(S_B)

        # 3. Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # 4. Sort
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # 5. Store Linear Discriminants
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        # Project data
        return np.dot(X, self.linear_discriminants.T)


if __name__ == "__main__":
    # Example Usage
    # 2 features, 2 classes
    X = np.array([
        [1, 2], [2, 3], [3, 3], [4, 5], [5, 5], # Class 0
        [1, 0], [2, 1], [3, 1], [3, 0], [2, 0]  # Class 1 (shifted down)
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    lda = LDA(n_components=1)
    lda.fit(X, y)
    X_projected = lda.transform(X)

    print("LDA Project from Scratch")
    print("Original Shape:", X.shape)
    print("Projected Shape:", X_projected.shape)
    # Class separability should be maximized
