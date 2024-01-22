import numpy as np
from scipy import stats
import networkx as nx


def find_overpresented_patterns(Histograms_, Subgroup_ids_, HR, adjust=False):
    if adjust:
        alpha = 0.05 / len(HR)
    else:
        alpha = 0.05
    Overpresented_patterns = {}
    for hr_dict in HR:
        subgroup_id = hr_dict["subgroup_id"]
        if hr_dict["p"]:
            Histogram_intra_group = Histograms_[Subgroup_ids_ == subgroup_id]
            Overpresented_patterns[subgroup_id] = []
            for other_subgroup_id in [
                i for i in range(1, len(HR) + 1) if i != subgroup_id
            ]:
                Histogram_other_group = Histograms_[Subgroup_ids_ == other_subgroup_id]
                candidates = []
                for i in range(Histograms_.shape[1]):
                    rvsi = Histogram_intra_group[:, i]
                    rvso = Histogram_other_group[:, i]
                    test_result = stats.mannwhitneyu(rvsi, rvso, alternative="greater")
                    if (test_result.pvalue < 0.05/Histograms_.shape[1]) & (np.median(rvsi) > 0.05):
                        candidates.append(i)
                        # print(
                        #     "Pattern {} overpresented in S{} by {:.3f} times (p={:.5f})".format(
                        #         i,
                        #         subgroup_id,
                        #         np.mean(rvsi) / np.mean(rvso),
                        #         test_result.pvalue,
                        #         # existence_intra_group * 100,
                        #         # existence_out_group * 100,
                        #     )
                        # )
                Overpresented_patterns[subgroup_id].append(set(candidates))
            Overpresented_patterns[subgroup_id] = list(
                set.intersection(*Overpresented_patterns[subgroup_id])
            )

    return Overpresented_patterns

# def find_overpresented_patterns(Histograms_, Subgroup_ids_, HR, adjust=False):
#     if adjust:
#         alpha = 0.05 / len(HR)
#     else:
#         alpha = 0.05
#     Overpresented_patterns = {}
#     for hr_dict in HR:
#         subgroup_id = hr_dict["subgroup_id"]
#         if hr_dict["p"]:
#             Overpresented_patterns[subgroup_id] = []
#             Histogram_intra_group = Histograms_[Subgroup_ids_ == subgroup_id]
#             Histogram_other_group = Histograms_[Subgroup_ids_ != subgroup_id]
#             for i in range(Histograms_.shape[1]):
#                 rvsi = Histogram_intra_group[:, i]
#                 rvso = Histogram_other_group[:, i]
#                 test_result = stats.mannwhitneyu(rvsi, rvso, alternative="greater")
#                 if (test_result.pvalue < 0.05):
#                     Overpresented_patterns[subgroup_id].append(i)

#     return Overpresented_patterns

def find_underpresented_patterns(Histograms_, Subgroup_ids_, HR, adjust=False):
    if adjust:
        alpha = 0.05 / len(HR)
    else:
        alpha = 0.05
    Overpresented_patterns = {}
    for hr_dict in HR:
        subgroup_id = hr_dict["subgroup_id"]
        if hr_dict["p"]:
            Histogram_intra_group = Histograms_[Subgroup_ids_ == subgroup_id]
            Overpresented_patterns[subgroup_id] = []
            for other_subgroup_id in [
                i for i in range(1, len(HR) + 1) if i != subgroup_id
            ]:
                Histogram_other_group = Histograms_[Subgroup_ids_ == other_subgroup_id]
                candidates = []
                for i in range(Histograms_.shape[1]):
                    rvsi = Histogram_intra_group[:, i]
                    rvso = Histogram_other_group[:, i]
                    test_result = stats.mannwhitneyu(rvsi, rvso)
                    if (test_result.pvalue < 0.05/Histograms_.shape[1]) and (
                        np.median(rvsi) < np.median(rvso)
                    ):
                        candidates.append(i)
                        # print(
                        #     "Pattern {} overpresented in S{} by {:.3f} times (p={:.5f})".format(
                        #         i,
                        #         subgroup_id,
                        #         np.mean(rvsi) / np.mean(rvso),
                        #         test_result.pvalue,
                        #         # existence_intra_group * 100,
                        #         # existence_out_group * 100,
                        #     )
                        # )
                Overpresented_patterns[subgroup_id].append(set(candidates))
            Overpresented_patterns[subgroup_id] = list(
                set.intersection(*Overpresented_patterns[subgroup_id])
            )

    return Overpresented_patterns


def find_representative_examples(
    pattern_id, Centroids, FILE_NAMES, X, Cluster_identities, Indices, num_examples=1
):
    Examples = []
    subtree_signature = Centroids[pattern_id, :]
    subtree_candidates = np.where(np.array(Cluster_identities) == pattern_id)[0]
    subtree_candidates_distance_to_signature = np.linalg.norm(
        X[subtree_candidates] - subtree_signature.reshape((1, -1)),
        axis=1,
    )
    Subtree_root_global_idx = subtree_candidates[
        np.argsort(subtree_candidates_distance_to_signature)[:num_examples]
    ]
    for subtree_root_global_idx in Subtree_root_global_idx:
        file_name = FILE_NAMES[Indices[subtree_root_global_idx]]
        subtree_root_local_idx = (
            subtree_root_global_idx
            - np.where(Indices == Indices[subtree_root_global_idx])[0][0]
        )
        # --------------------------------------
        patient_id = int(file_name.split("_")[1])
        image_id = int(file_name.split("_")[3])
        Examples.append((patient_id, image_id, subtree_root_local_idx))
    return Examples


def decide_subtree_boundary(root_idx, Adj, iteration, boundary_weight_threshold=0.1):
    W = Adj.copy()
    for i in range(iteration):
        W = np.matmul(W, Adj)
    leaf_indices = list(np.where(W[root_idx, :] > boundary_weight_threshold)[0])
    return leaf_indices


def construct_cellular_graph(
    Adj, cells_, cell_type, cell_type_id, cell_type_color, edges_visible=0.1
):
    Adj_ = Adj.copy()
    Adj_[Adj_ < edges_visible] = 0
    np.fill_diagonal(Adj_, 0)
    cellular_graph = nx.from_numpy_array(Adj_)
    cells_["cell_type"] = cell_type
    cells_["cell_type_id"] = cell_type_id
    cells_["cell_type_color"] = cell_type_color
    nx.set_node_attributes(cellular_graph, cells_["X"], "X")
    nx.set_node_attributes(cellular_graph, cells_["Y"], "Y")
    nx.set_node_attributes(cellular_graph, cells_["cell_type"], "cell_type")
    nx.set_node_attributes(cellular_graph, cells_["cell_type_id"], "cell_type_id")
    nx.set_node_attributes(cellular_graph, cells_["cell_type_color"], "cell_type_color")
    return cellular_graph


def calculate_relative_presentation(Histogram, Subgroup_ids, threshold):
    """
    Find patterns that are overpresented in the subgroup compared to the whole population.
    Parameters
    ----------
    Histogram : ndarray
        Histogram of the patterns of each patient. shape = (n_patients, n_patterns)
    Subgroup_ids : ndarray
        Ids of the patients in the subgroup. shape = (n_patients)
    threshold : float
        Threshold to consider a pattern overpresented in the subgroup.
    Returns
    -------
    overpresented_patterns : dict
        List of the patterns that are overpresented in the subgroup.
    """
    # Find the patterns that are overpresented in the subgroup compared to the whole population.
    median_presentation = np.median(Histogram, axis=0)
    Relative_presentation = {}
    unique_subgroup_ids = np.unique(Subgroup_ids[Subgroup_ids != 0])
    for subgroup_id in unique_subgroup_ids:
        Histogram_intra_group = Histogram[Subgroup_ids == subgroup_id]
        median_presentation_intra_group = np.median(Histogram_intra_group, axis=0)
        relative_presentation = median_presentation_intra_group / median_presentation
        Relative_presentation[subgroup_id] = relative_presentation
    return Relative_presentation
