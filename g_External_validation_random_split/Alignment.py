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
    Neighbors_labels = Label_reference[indices]  # m x k
    unique, indices_in_unique = np.unique(Neighbors_labels, return_inverse=True)
    label_query_hat = unique[
        np.argmax(
            np.apply_along_axis(
                np.bincount,
                1,
                indices_in_unique.reshape(Neighbors_labels.shape),  # m * len(unique)
                weights=None,
                minlength=np.max(indices_in_unique) + 1,
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
    Centroid_reference = np.zeros(
        (np.unique(Label_reference).shape[0], X_reference.shape[1])
    )
    for i in range(np.unique(Label_reference).shape[0]):
        Centroid_reference[i, :] = np.mean(X_reference[Label_reference == i, :], axis=0)
    # distance: m x k
    # indices: m x k
    neigh = NearestNeighbors(n_neighbors=1, radius=1)
    neigh.fit(Centroid_reference)
    distances, index = neigh.kneighbors(X_query)
    label_query_hat = index  # m x k
    return label_query_hat


from skimage import filters


def centroid_alignment_based_on_similarity(Similarity, Label_reference, k=10, quality_control_threshold=0.5):
    threshold = filters.threshold_otsu(Similarity)
    threshold = np.percentile(Similarity, 90)
    print(threshold)
    assert Similarity.shape[1] == len(Label_reference)
    Unique_label_reference = np.unique(Label_reference)
    Label_query_hat = np.zeros((Similarity.shape[0], len(Unique_label_reference)))
    # for i in range(len(Unique_label_reference)):
    #     unique_label = Unique_label_reference[i]
    #     similarity_to_unique_label = np.mean(
    #         Similarity[:, Label_reference == unique_label], axis=1
    #     )
    #     # otsu thresholding
    #     #threshold = filters.threshold_otsu(similarity_to_unique_label)
    #     Label_query_hat[similarity_to_unique_label > threshold, i] = 1
    # return Label_query_hat
    # print(Unique_label_reference)
    # Similarity_group = np.zeros((Similarity.shape[0], len(Unique_label_reference)))
    # for i in range(len(Unique_label_reference)):
    #     Similarity_group[:, i] = np.median(np.sort(Similarity[:, Label_reference == Unique_label_reference[i]], axis=1)[:, -k:], axis=1)
    
    # # Label_query_hat = Unique_label_reference[np.argmax(Similarity_group, axis=1)]
    # # print(np.argmax(Similarity_group, axis=1), np.argmax(Similarity_group, axis=1).shape, Label_query_hat.shape)
    # print(np.max(Similarity_group, axis=1) )
    # for i in range(len(Unique_label_reference)):
        
    #     Label_query_hat[(np.argmax(Similarity_group, axis=1) == i) & (np.max(Similarity_group, axis=1) >quality_control_threshold), i] = 1
    for i in range(Similarity.shape[0]):
        label_knn = Label_reference[np.argsort(Similarity[i, :])[::-1][:k]]
        if len(np.unique(label_knn)) == 1:
            Label_query_hat[i, int(np.unique(label_knn)-1)] = 1
        # Label_query_hat[i, np.argmax(Similarity[i, :])] = 1
    return Label_query_hat
