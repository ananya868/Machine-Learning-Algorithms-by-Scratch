"""Gaussian Naive Bayes"""
# Classification algorithm based on Bayes' Theorem with an assumption of independence among predictors.
# Gaussian Naive Bayes assumes that the continuous values associated with each class are distributed according to a Gaussian distribution.
# P(y|X) = (P(X|y) * P(y)) / P(X)
# - P(y|X): Posterior probability
# - P(X|y): Likelihood (Gaussian PDF)
# - P(y): Prior probability
# - P(X): Evidence (Constant for all classes, so ignored in maximization)

import math

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X 
        self.y = y 
        self.n_samples = len(X)
        self.n_features = len(X[0])
        self.classes = sorted(list(set(y)))
        self.n_classes = len(self.classes)

        # Calculate mean, variance, and prior for each class
        self._mean = [[0.0] * self.n_features for _ in range(self.n_classes)]
        self._var = [[0.0] * self.n_features for _ in range(self.n_classes)]
        self._priors = [0.0] * self.n_classes

        for idx, c in enumerate(self.classes):
            # Filter samples for the current class
            # X_c = [row for i, row in enumerate(X) if y[i] == c]
            # Since we are avoiding numpy for filtering, we do it manually with indices
            X_c_indices = [i for i, label in enumerate(y) if label == c]
            X_c = [X[i] for i in X_c_indices]
            
            # Update mean, var, prior
            self._mean[idx] = self._calculate_mean(X_c)
            self._var[idx] = self._calculate_var(X_c)
            self._priors[idx] = len(X_c) / self.n_samples

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = math.log(self._priors[idx])
            class_conditional = sum(math.log(self._pdf(idx, feature_idx, x[feature_idx])) for feature_idx in range(self.n_features))
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        return self.classes[posteriors.index(max(posteriors))]

    def _pdf(self, class_idx, feature_idx, x):
        # Gaussian Probability Density Function
        mean = self._mean[class_idx][feature_idx]
        var = self._var[class_idx][feature_idx]
        numerator = math.exp(-((x - mean) ** 2) / (2 * var))
        denominator = math.sqrt(2 * math.pi * var)
        return numerator / denominator

    def _calculate_mean(self, X):
        # Mean of each feature column
        n = len(X)
        features = len(X[0])
        means = []
        for j in range(features):
            col_sum = sum(row[j] for row in X)
            means.append(col_sum / n)
        return means

    def _calculate_var(self, X):
        # Variance of each feature column
        n = len(X)
        features = len(X[0])
        means = self._calculate_mean(X)
        vars = []
        for j in range(features):
            variance = sum((row[j] - means[j]) ** 2 for row in X) / n
            vars.append(variance)
        return vars

if __name__ == "__main__":
    # Example Usage
    # Features: [height (cm), weight (kg), foot size (cm)]
    X_train = [
        [170, 70, 26], [172, 75, 27], [165, 65, 25], # Males
        [150, 45, 22], [155, 50, 23], [160, 52, 24]  # Females
    ]
    y_train = [0, 0, 0, 1, 1, 1] # 0: Male, 1: Female

    X_test = [
        [168, 72, 26], # Likely Male
        [152, 48, 22]  # Likely Female
    ]

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Gaussian Naive Bayes Classification from Scratch")
    print(f"Predictions: {predictions}")
    # Expected: [0, 1]
