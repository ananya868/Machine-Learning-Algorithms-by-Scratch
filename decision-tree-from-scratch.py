"""Decision Tree Classifier"""
# Supervised learning algorithm used for classification and regression.
# Builds a flowchart-like tree structure using simple decision rules inferred from the data features.
# Segments the data into subsets based on the value of input features.
# Uses 'Entropy' and 'Information Gain' to determine the best split.

import math
import collections

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
        # Convert to list of lists if not already (for consistency)
        self.n_features = len(X[0]) if not self.n_features else min(self.n_features, len(X[0]))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = len(X)
        n_feats = len(X[0])
        n_labels = len(collections.Counter(y))

        # Check the stopping criteria
        # 1. Max depth reached
        # 2. Only 1 class label left
        # 3. Not enough samples to split
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = list(range(n_feats))
        # (Could shuffle feat_idxs here if we wanted stochastic behavior/Random Forest prep)
        
        # Find the best split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        # If no split improves information gain, make it a leaf
        if best_feat is None:
             leaf_value = self._most_common_label(y)
             return Node(value=leaf_value)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feat] if hasattr(X, 'shape') else [x[best_feat] for x in X], best_thresh)
        
        # Helper to slice X and y based on indices (manual list slicing)
        left_X = [X[i] for i in left_idxs]
        left_y = [y[i] for i in left_idxs]
        right_X = [X[i] for i in right_idxs]
        right_y = [y[i] for i in right_idxs]

        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)
        
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_column = [x[feat_idx] for x in X]
            thresholds = set(X_column)
            
            for thr in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, X_column, thr)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
                    
        if best_gain == -1: # No split occurred 
             return None, None

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)
        
        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy([y[i] for i in left_idxs]), self._entropy([y[i] for i in right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        # calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = [i for i, x in enumerate(X_column) if x <= split_thresh]
        right_idxs = [i for i, x in enumerate(X_column) if x > split_thresh]
        return left_idxs, right_idxs

    def _entropy(self, y):
        # E = -Sum(p(x) * log2(p(x)))
        hist = collections.Counter(y)
        ps = [count / len(y) for count in hist.values()]
        return -sum(p * math.log2(p) for p in ps if p > 0)

    def _most_common_label(self, y):
        counter = collections.Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

if __name__ == "__main__":
    # Example usage
    X_train = [
        [2, 3], [3, 3], [1, 1],
        [6, 6], [7, 7], [5, 5],
        [6, 3], [2, 6]
    ]
    # 0 = close to origin, 1 = far, 2 = mixed? Let's make logical labels
    # If sum < 10 -> 0, else -> 1
    y_train = [0, 0, 0, 1, 1, 1, 0, 0] # [2,6] sum is 8 -> 0
    
    X_test = [[2, 2], [7, 6], [8, 8]]
    
    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    print("Decision Tree Classification from Scratch")
    print(f"Predictions: {predictions}")
    # Expected: [0, 1, 1]
