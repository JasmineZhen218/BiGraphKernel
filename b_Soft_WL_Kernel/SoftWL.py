import numpy as np
import phenograph
from sklearn.metrics import pairwise_distances
import os


def get_Gram_matrix(Histograms):
    """
    parameters:
    -----------
    Histograms: list of histograms of TME patterns
    return:
    -----------
    SoftWL kernel Gram matrix
    """
    Gram_matrix = np.zeros((len(Histograms), len(Histograms)))
    for i in range(len(Histograms)):
        for j in range(i, len(Histograms)):
            Gram_matrix[i, j] = SoftWL_kernel(Histograms[i], Histograms[j])
            Gram_matrix[j, i] = Gram_matrix[i, j]
    return Gram_matrix


def SoftWL_kernel(histogram_i, histogram_j):
    k_ij = np.inner(histogram_i, histogram_j)
    denominator = np.sqrt(
        np.inner(histogram_i, histogram_i) * np.inner(histogram_j, histogram_j)
    )
    if denominator == 0:
        print("denominator is 0")
        print(f"histogram_i = {histogram_i}")
        print(f"histogram_j = {histogram_j}")
        print(f"k_ij = {k_ij}, divide {denominator}")
    k_ij_normalized = k_ij / denominator
    return k_ij_normalized


def neighborhood_aggregation(x0, adj, iteration):
    """
    neighborhood aggregation
    parameters:
    -----------
    x0: initial node labels
    adj: adjacency matrix
    iteration: number of iterations

    return:
    ---------
    x: node labels after neighborhood aggregation
    """
    assert adj.diagonal()[0] == 1
    x = x0
    for iter in range(iteration):
        x = adj @ x
    return x


def cluster_subtrees(X, k, n_job):
    """
    cluster subtrees
    parameters:
    -----------
    X: node labels after neighborhood aggregation

    return:
    ---------
    cluster: cluster labels of subtrees
    """
    # initial clustering
    Cluster_identities, _, _ = phenograph.cluster(X, n_jobs=n_job, k=k)
    return Cluster_identities


def compute_cluster_centroids(X, Cluster_identities):
    """
    compute cluster centroids
    parameters:
    -----------
    X: node labels after neighborhood aggregation
    Cluster_identities: cluster labels of subtrees

    return:
    ---------
    centroids: cluster centroids
    """
    centroids = []
    unique_clusters = np.unique(Cluster_identities)
    for cluster in unique_clusters:
        cluster_points = X[Cluster_identities == cluster]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


def merge_close_clusters(X, Cluster_identities, merge_threshold):
    """
    merge close clusters
    parameters:
    -----------
    X: node labels after neighborhood aggregation
    Cluster_identities: cluster labels of subtrees
    merge_threshold: threshold to merge clusters

    return:
    ---------
    Cluster_identities: cluster labels of subtrees after merging
    """
    centroids = compute_cluster_centroids(X, Cluster_identities)
    pairwise_dist = pairwise_distances(centroids)
    close_pairs_mask = pairwise_dist < merge_threshold
    # connected components of close pairs

    Cluster_identities_merged = Cluster_identities.copy()
    num_clusters = len(np.unique(Cluster_identities))
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            if pairwise_dist[i, j] < merge_threshold:
                Cluster_identities_merged[Cluster_identities_merged == j] = i
    return Cluster_identities_merged
