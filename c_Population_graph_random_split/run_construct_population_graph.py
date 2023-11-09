import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from population_graph import construct_PopulationGraph

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iteration", type=int, default=2, help="Iteration of neighborhood aggregation"
)
parser.add_argument(
    "--PhenoGraph_k",
    type=int,
    default=100,
    help="Neighbor of neighborhood in PhenoGraph",
)
parser.add_argument(
    "--PopulationGraph_type",
    type=str,
    default="complete_graph_with_weak_edges_removed",
    help="Type of Population Graph",
)
parser.add_argument(
    "--weight_threshold_percentile",
    type=int,
    default=95,
    help="weight threshold percentile for weak edges removal",
)
parser.add_argument(
    "--knn_k",
    type=int,
    default=10,
    help="Number of nearest neighbors for constructing knn graph",
)
args = parser.parse_args()
print(args)

para_dict = {
    "weight_threshold_percentile": args.weight_threshold_percentile,
    "knn_k": args.knn_k,
}

print("Loading SoftWL_dict...")
SoftWL_dict = pickle.load(
    open(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "SoftWL_dict_iter_"
            + str(args.iteration)
            + "_PhenoGraph_k_"
            + str(args.PhenoGraph_k)
            + ".pkl",
        ),
        "rb",
    ),
)
Patient_IDs = SoftWL_dict["Patient_id"]
Gram_matrix = SoftWL_dict["Gram_matrix"]
Histograms = SoftWL_dict["Histogram"]

print("Constructing Population Graph...")
G_population = construct_PopulationGraph(
    Gram_matrix,
    args.PopulationGraph_type,
    para_dict=para_dict,
)

print("Ploting and saving Population Graph...")
pos = nx.spring_layout(G_population, seed=2, k=1 / np.sqrt(682) * 5, iterations=100)
f, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
edge_list = list(G_population.edges())
edge_alpha = [
    0.1 * G_population[u][v]["weight"] if G_population[u][v]["weight"] > 0 else 0
    for u, v in edge_list
]
nx.draw_networkx_edges(G_population, pos, alpha=edge_alpha, width=2)
nx.draw_networkx_nodes(
    G_population, pos, node_size=80, node_color="white", edgecolors="black"
)
handles = []
patch = Line2D(
    [0],
    [0],
    marker="o",
    color="black",
    label=f"patient (N={len(G_population.nodes)})",
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=1,
    markersize=8,
)
handles.append(patch)
ax.legend(handles=handles)
os.makedirs(
    os.path.join(
        PROJECT_ROOT,
        "Output",
        "c_Population_graph",
        "Danenberg",
        "Cohort_1",
        "iter_" + str(args.iteration) + "_PhenoGraph_k_" + str(args.PhenoGraph_k),
        args.PopulationGraph_type,
    ),
    exist_ok=True,
)
if args.PopulationGraph_type == "complete_graph":
    f.savefig(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "c_Population_graph",
            "Danenberg",
            "Cohort_1",
            "iter_" + str(args.iteration) + "_PhenoGraph_k_" + str(args.PhenoGraph_k),
            args.PopulationGraph_type,
            "Population_graph.png",
        )
    )
elif args.PopulationGraph_type == "complete_graph_with_weak_edges_removed":
    f.savefig(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "c_Population_graph",
            "Danenberg",
            "Cohort_1",
            "iter_" + str(args.iteration) + "_PhenoGraph_k_" + str(args.PhenoGraph_k),
            args.PopulationGraph_type,
            "Population_graph_weight_threshold_percentile_"
            + str(args.weight_threshold_percentile)
            + ".png",
        )
    )
elif args.PopulationGraph_type in ["knn_graph", "two_step_knn_graph"]:
    f.savefig(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "c_Population_graph",
            "Danenberg",
            "Cohort_1",
            "iter_" + str(args.iteration) + "_PhenoGraph_k_" + str(args.PhenoGraph_k),
            args.PopulationGraph_type,
            "Population_graph_knn_k_" + str(args.knn_k) + ".png",
        )
    )


