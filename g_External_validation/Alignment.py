import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_alignment(X_reference, Label_reference, X_query, k=100):
    # feature_reference: n x d
    # label_reference: n x 1
    # feature_query: m x d
    # label_query_estimated: m x 1
    neigh = NearestNeighbors(n_neighbors=k, radius=1)
    neigh.fit(X_reference)
    distances, indices = neigh.kneighbors(X_query) 
    # distance: m x k
    # indices: m x k
    Neighbors_labels = Label_reference[indices] # m x k
    unique, indices_in_unique = np.unique(Neighbors_labels, return_inverse=True) 
    label_query_hat = unique[
        np.argmax(
            np.apply_along_axis(
                np.bincount,
                1,
                indices_in_unique.reshape(Neighbors_labels.shape), # m * len(unique)
                weights = None,
                minlength = np.max(indices_in_unique) + 1,
            ),
            axis=1,
        )
    ]
    return label_query_hat


def centroid_alignment(X_reference, Label_reference, X_query, k=100):
    # feature_reference: n x d
    # label_reference: n x 1
    # feature_query: m x d
    # label_query_estimated: m x 1
    Centroid_reference = np.zeros((np.unique(Label_reference).shape[0], X_reference.shape[1]))
    for i in range(np.unique(Label_reference).shape[0]):
        Centroid_reference[i, :] = np.mean(X_reference[Label_reference == i, :], axis=0)
    # distance: m x k
    # indices: m x k
    neigh = NearestNeighbors(n_neighbors=1, radius=1)
    neigh.fit(Centroid_reference)
    distances, index = neigh.kneighbors(X_query) 
    label_query_hat = index # m x k
    return label_query_hat