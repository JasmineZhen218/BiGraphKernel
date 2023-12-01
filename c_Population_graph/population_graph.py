import numpy as np
import networkx as nx

def construct_PopulationGraph(Gram_matrix, PopulationGraph_type, para_dict={}):
    """ "
    Construct Population graph from gram matrix
    Parameters
    ----------
    Gram_matrix: numpy array
        Gram_matrix matrix
    PopulationGraph_type: str
        "complete_graph", "complete_graph_with_weak_edge_removed", "knn_graph", "two_step_knn_graph"
    para_dict: dict
        parameters for PhenoGraph construction
    Returns
    -------
    G_population: networkx graph
    """
    Gram_matrix_ = Gram_matrix.copy()
    np.fill_diagonal(Gram_matrix_, 0)
    if PopulationGraph_type == "complete_graph":
        G_population = nx.from_numpy_array(Gram_matrix_)
    elif PopulationGraph_type == "complete_graph_with_weak_edges_removed":
        if 'weight_threshold_percentile' in para_dict.keys():
            Gram_matrix_[Gram_matrix_ < np.percentile(Gram_matrix, para_dict['weight_threshold_percentile'])] = 0
        elif 'weight_threshold' in para_dict.keys():
            Gram_matrix_[Gram_matrix_ < para_dict['weight_threshold']] = 0
        G_population = nx.from_numpy_array(Gram_matrix_)
    elif PopulationGraph_type == "knn_graph":
        for i in range(Gram_matrix_.shape[0]):
            Gram_matrix_[i, np.argsort(Gram_matrix_[i, :])[: -para_dict["knn_k"]]] = 0
        Gram_matrix_ = np.maximum(Gram_matrix_, Gram_matrix_.transpose()) # make it symmetric
        G_population = nx.from_numpy_array(Gram_matrix_)
    elif PopulationGraph_type == "two_step_knn_graph":
        for i in range(Gram_matrix_.shape[0]):
            Gram_matrix_[i, np.argsort(Gram_matrix_[i, :])[: -para_dict["knn_k"]]] = 0
        adj_1 = np.maximum(Gram_matrix_, Gram_matrix_.transpose()) # make it symmetric
        adj_2 = np.zeros_like(Gram_matrix_)
        for i in range(Gram_matrix_.shape[0]):
            for j in range(Gram_matrix_.shape[0]):
                neighbor_i = np.where(adj_1[i, :] > 0)[0]
                neighbor_j = np.where(adj_1[j, :] > 0)[0]
                if len(
                    set(neighbor_i).union(set(neighbor_j))
                ) == 0:
                    weight = 0
                else:
                    weight = len(set(neighbor_i).intersection(set(neighbor_j))) / len(
                        set(neighbor_i).union(set(neighbor_j))
                    )
                adj_2[i, j] = weight
        np.fill_diagonal(adj_2, 0)
        G_population = nx.from_numpy_array(adj_2)
    return G_population