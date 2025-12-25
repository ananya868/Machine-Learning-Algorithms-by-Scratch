"""Random Forest Classifier"""
# Ensemble learning method for classification (can be extended to regression).
# Constructs a multitude of decision trees at training time.
# Output class is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
# Key Concepts:
# 1. Bootstrapping: Random sampling with replacement.
# 2. Feature Randomness: Random subset of features considered for splitting.

import collections
import math
import random

# Reuse Decision Tree logic (Simplified for inclusion)
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = len(X[0]) if not self.n_features else min(self.n_features, len(X[0]))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = len(X)
        n_feats = len(X[0])
        n_labels = len(collections.Counter(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = random.sample(range(n_feats), self.n_features)
        
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
             leaf_value = self._most_common_label(y)
             return Node(value=leaf_value)

        left_idxs, right_idxs = self._split([x[best_feat] for x in X], best_thresh)
        
        left = self._grow_tree([X[i] for i in left_idxs], [y[i] for i in left_idxs], depth + 1)
        right = self._grow_tree([X[i] for i in right_idxs], [y[i] for i in right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = [x[feat_idx] for x in X]
            thresholds = set(X_column)
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy([y[i] for i in left_idxs]), self._entropy([y[i] for i in right_idxs])
        return parent_entropy - ((n_l / n) * e_l + (n_r / n) * e_r)

    def _split(self, X_column, split_thresh):
        left_idxs = [i for i, x in enumerate(X_column) if x <= split_thresh]
        right_idxs = [i for i, x in enumerate(X_column) if x > split_thresh]
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = collections.Counter(y)
        ps = [count / len(y) for count in hist.values()]
        return -sum(p * math.log2(p) for p in ps if p > 0)

    def _most_common_label(self, y):
        counter = collections.Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold: return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = len(X)
        idxs = [random.choice(range(n_samples)) for _ in range(n_samples)]
        return [X[i] for i in idxs], [y[i] for i in idxs]

    def predict(self, X):
        tree_preds = [tree.predict(X) for tree in self.trees]
        # Transpose: from (n_trees, n_samples) to (n_samples, n_trees)
        # tree_preds = [[preds_tree1], [preds_tree2]] -> [[sample1_tree1, sample1_tree2], ...]
        tree_preds = list(map(list, zip(*tree_preds)))
        
        predictions = []
        for preds in tree_preds:
            # Majority vote
            most_common = collections.Counter(preds).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions

if __name__ == "__main__":
    X_train = [[1, 2], [2, 1], [2, 3], [5, 4], [6, 5], [5, 6]]
    y_train = [0, 0, 0, 1, 1, 1]
    X_test = [[1, 1], [5, 5]]

    clf = RandomForest(n_trees=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    print("Random Forest from Scratch")
    print(f"Predictions: {predictions}")
