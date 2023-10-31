import networkx as nx
import numpy as np

def detect_communities(G, size_smallest_cluster):
    """
    Detect communities in a graph
    Parameters
    ----------
    G: networkx graph
        graph
    size_smallest_cluster: int
        size of the smallest cluster
    Returns
    -------
    Community_ids: array
        
    """
    Communities = nx.community.louvain_communities(
                G, weight="weight", resolution=1, seed=1
            )
    Communities = [list(c) for c in Communities if len(c) > size_smallest_cluster]
    Communities = sorted(Communities, key=lambda x: len(x), reverse=True)
    Community_ids = np.zeros(G.number_of_nodes())
    for i, c in enumerate(Communities):
        Community_ids[np.array(c)] = i+1
    return Community_ids