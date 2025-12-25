"""K-Nearest Neighbors (KNN)"""
# Supervised learning algorithm used for both classification and regression.
# Non-parametric and lazy learning algorithm.
# - Non-parametric: No assumption for underlying data distribution.
# - Lazy learning: Does not need any training data points for model generation (no training phase).
# Relies on distance (e.g., Euclidean) between feature vectors.

import collections
import math

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # KNN is a lazy learner, so we just store the training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = collections.Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        # sqrt(sum((x1 - x2)^2))
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

if __name__ == "__main__":
    # Example usage
    
    # Simple 2D dataset
    # Class 0: Points usually around (1,1)
    # Class 1: Points usually around (5,5)
    X_train = [[1, 2], [2, 1], [2, 3], [5, 4], [6, 5], [5, 6]]
    y_train = [0, 0, 0, 1, 1, 1]

    # Test points
    X_test = [[1, 1], [5, 5], [3, 3]]
    
    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    print("KNN Classification from Scratch")
    print(f"Train X: {X_train}")
    print(f"Train y: {y_train}")
    print(f"Test X: {X_test}")
    print(f"Predictions (k={k}): {predictions}")
    # Expected: [0, 1, 0] or similar depending on the boundary
