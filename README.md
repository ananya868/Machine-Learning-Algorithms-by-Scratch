# Machine Learning Algorithms From Scratch

A collection of basic Machine Learning algorithms implemented in pure Python, without using data science libraries like NumPy, Pandas, or Scikit-Learn.

This repository serves as an educational resource to help understand the mathematics and logic behind these algorithms by building them from the ground up.

## Implemented Algorithms

### Supervised Learning
*   **Linear Regression**: Implements both Ordinary Least Squares (OLS) and Gradient Descent optimization.
*   **Logistic Regression**: For binary classification tasks.
*   **Polynomial Regression**: Fits non-linear relationships.
*   **Decision Tree**: Implementation of decision tree for classification.
*   **Random Forest**: Ensemble method using multiple decision trees.
*   **Naive Bayes**: Probabilistic classifier based on Bayes' theorem.
*   **Support Vector Machines (SVM)**: Finds the optimal hyperplane for classification.
*   **K-Nearest Neighbors (KNN)**: Classification based on feature similarity.
*   **Adaboost**: Boosting ensemble technique.
*   **Linear Discriminant Analysis (LDA)**: For dimensionality reduction and classification.
*   **Neural Network**: Basic implementation of a neural network.

### Unsupervised Learning
*   **K-Means Clustering**: Partitioning data into K distinct clusters.
*   **Principal Component Analysis (PCA)**: Dimensionality reduction technique.

## Getting Started

These implementations rely solely on Python's standard library (modules like `math`, `random`, `collections`). No external dependencies are required.

### Prerequisites

*   Python 3.x

### Running the Algorithms

Each file is self-contained and includes a `__main__` block with sample data and example usage. You can run any algorithm directly from the terminal.

**Example: Running Linear Regression**

```bash
python linear-regression-from-scratch.py
```

**Output:**

```text
::::Using OLS Method::::
Slope: 0.8 
Intercept: 1.6
Prediction for x = 6: 6.4

::::Using Gradient Descent Method::::
Slope: 0.8 
Intercept: 1.6
Prediction for x = 6: 6.4
```

## Structure

*   Every file follows the pattern `[algorithm-name]-from-scratch.py`.
*   The code is commented with explanations of the mathematical concepts used (e.g., cost functions, gradient updates).