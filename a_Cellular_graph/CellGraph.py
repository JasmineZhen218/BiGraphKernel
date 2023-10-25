from scipy.spatial.distance import pdist, squareform
import numpy as np


    


def Pos2Adj(Pos, a):
    """
    Compute the adjacency matrix of a graph from the position of its nodes. 
    The edge weight is exp(- a * distance^2).
    Parameters
    ----------
    Pos : array_like
        Position of the nodes in the graph.
    a : float
        Parameter of the edge weight.
    Returns
    ------- 
    Adj : array_like
        Adjacency matrix of the graph. (Diagonal elements are one.)
    """
    Distance = squareform(pdist(Pos)) # Euclidean distance between all pairs of points
    Adj = np.exp(- a* Distance * Distance) # edge weight = exp(- a * distance^2)
    return Adj


