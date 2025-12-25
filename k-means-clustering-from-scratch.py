"""K-Means Clustering"""
# Unsupervised learning algorithm to partition data into k clusters.
# Iterative algorithm:
# 1. Initialize k centroids randomly.
# 2. Assign each data point to the nearest centroid.
# 3. Update centroids by calculating the mean of all points assigned to that cluster.
# 4. Repeat until convergence.

import math
import random

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def fit(self, X):
        self.X = X
        self.n_samples = len(X)
        self.n_features = len(X[0])

        # Initialize centroids
        # Ideally, we should pick random samples from X as initial centroids
        random_sample_idxs = random.sample(range(self.n_samples), self.K)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check for convergence
            if self._is_converged(centroids_old, self.centroids):
                break

        return self.clusters, self.centroids

    def predict(self, X):
        clusters = self._create_clusters(self.centroids)
        # Return cluster label for each sample
        labels = [0] * self.n_samples
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [self._euclidean_distance(sample, point) for point in centroids]
        closest_index = distances.index(min(distances))
        return closest_index

    def _get_centroids(self, clusters):
        # Assign mean value of clusters to centroids
        centroids = [[0.0] * self.n_features for _ in range(self.K)]
        for cluster_idx, cluster in enumerate(clusters):
            # Mean of features
            # Calculate mean feature vector for the cluster
            if not cluster: # Handle empty cluster
                # Keep old centroid or re-initialize (Here we just keep old if possible, or 0)
                # Ideally, should handle better, but for scratch implementation:
                continue 
                
            cluster_mean = [0.0] * self.n_features
            for sample_idx in cluster:
                sample = self.X[sample_idx]
                for i in range(self.n_features):
                    cluster_mean[i] += sample[i]
            
            centroids[cluster_idx] = [x / len(cluster) for x in cluster_mean]
            
        return centroids

    def _is_converged(self, centroids_old, centroids_new):
        # Distances between each old and new centroids, sum all
        distances = [self._euclidean_distance(centroids_old[i], centroids_new[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _euclidean_distance(self, x1, x2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

if __name__ == "__main__":
    # Example Data
    X = [
        [1, 2], [1, 4], [1, 0], # Cluster 1
        [10, 2], [10, 4], [10, 0] # Cluster 2
    ]
    
    k = 2
    kmeans = KMeans(K=k, max_iters=100)
    clusters, centroids = kmeans.fit(X)
    
    print("K-Means Clustering from Scratch")
    print(f"Final Centroids: {centroids}")
    
    # Simple check: First 3 should be in one cluster, next 3 in another
    pred_labels = kmeans.predict(X)
    print(f"Predicted Cluster Labels: {pred_labels}")
