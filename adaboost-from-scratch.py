"""AdaBoost Classifier"""
# Component of Ensemble Learning.
# Adaptive Boosting: Weak learners are combined to form a strong learner.
# Use Decision Stumps (Tree with depth 1) as weak learners.
# Each stump tries to correct the errors of the "previous" stump.
# Sample weights are updated (increased for misclassified, decreased for correctly classified).

import math

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = len(X)
        X_column = [x[self.feature_idx] for x in X]
        predictions = [1] * n_samples
        
        if self.polarity == 1:
            predictions = [-1 if x < self.threshold else 1 for x in X_column]
        else:
            predictions = [1 if x < self.threshold else -1 for x in X_column]
            
        return predictions

class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize weights to 1/N
        w = [1 / n_samples] * n_samples
        
        # Convert 0/1 to -1/1 if necessary (AdaBoost standard)
        y_ = [-1 if lbl == 0 else 1 for lbl in y]

        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            
            min_error = float('inf')
            
            # Greedy search to find best threshold and feature
            for feature_i in range(n_features):
                X_column = [x[feature_i] for x in X]
                thresholds = sorted(list(set(X_column)))
                
                for threshold in thresholds:
                    # Check both polarities (less than or greater than)
                    for polarity in [1, -1]:
                        predictions = [1] * n_samples
                        if polarity == 1:
                            predictions = [-1 if x < threshold else 1 for x in X_column]
                        else:
                            predictions = [1 if x < threshold else -1 for x in X_column]

                        # Weighted error
                        error = sum(w[i] for i in range(n_samples) if y_[i] != predictions[i])
                        
                        if error < min_error:
                            min_error = error
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_idx = feature_i

            # Calculate alpha (Influence of this classifier)
            # alpha = 0.5 * log((1-e)/e)
            EPS = 1e-10
            clf.alpha = 0.5 * math.log((1 - min_error + EPS) / (min_error + EPS))

            # Update weights
            # w = w * exp(-alpha * y * prediction)
            predictions = clf.predict(X)
            w = [w[i] * math.exp(-clf.alpha * y_[i] * predictions[i]) for i in range(n_samples)]
            
            # Normalize weights
            sum_w = sum(w)
            w = [val / sum_w for val in w]

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.predict(X) for clf in self.clfs]
        
        # Weighted sum: Sum(alpha * prediction)
        # Transpose to get preds per sample
        n_samples = len(X)
        y_pred = []
        
        for i in range(n_samples):
            # Sum of alphas * predictions for sample i
            s = sum(clf.alpha * clf_preds[j][i] for j, clf in enumerate(self.clfs))
            y_pred.append(s)
            
        # Sign of the sum
        return [1 if s >= 0 else 0 for s in y_pred]

if __name__ == "__main__":
    # Example
    X = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]]
    y = [0, 0, 1, 1, 1] # 0 becomes -1

    clf = AdaBoost(n_clf=5)
    clf.fit(X, y)
    preds = clf.predict(X)

    print("AdaBoost from Scratch")
    print(f"Predictions: {preds}")
