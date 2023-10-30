import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
from SoftWL import cluster_subtrees
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_job", type=int, default=-1, help="Number of jobs for parallel computing"
)
parser.add_argument(
    "--iteration", type=int, default=1, help="Iteration of neighborhood aggregation"
)
parser.add_argument(
    "--k", type=int, default=30, help="Neighbor of neighborhood in PhenoGraph"
)
args = parser.parse_args()


FILE_NAMES = os.listdir(
    os.path.join(
        PROJECT_ROOT, "Output", "b_Soft_WL_Kernel", "Danenberg", "Cohort_1", "Subtrees"
    )
)
assert len(FILE_NAMES) == 467
# Concatenate all subtrees from all cellular graphs
X_concat = []
Indices = []
for i in range(len(FILE_NAMES)):
    file_name = FILE_NAMES[i]
    patient_id = int(file_name.split("_")[1])
    X = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X" + str(args.iteration) + ".npy",
        )
    )
    X_concat.append(X)
    indices = np.zeros(X.shape[0], dtype=int)
    indices[:] = i
    Indices.append(indices)
X_concat = np.concatenate(X_concat, axis=0)
Indices = np.concatenate(Indices, axis=0)
assert X_concat.shape[0] == Indices.shape[0]
print(
    f"Concatenated feature shape {X_concat.shape}",
    f"{X_concat.shape[0]} subtrees from {len(np.unique(Indices))} cellular graphs",
)

# Cluster all subtrees
# call "export OMP_NUM_THREADS=1" before running to avoid "Too many memory regions" error with Dask
Initial_Cluster_X = cluster_subtrees(X_concat, k=args.k, n_job=-1)
print("{} unique clusters".format(len(np.unique(Initial_Cluster_X))))
# Save initial clusters
for i in range(len(FILE_NAMES)):
    file_name = FILE_NAMES[i]
    os.makedirs(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Subtrees",
            "Cohort_1",
            file_name,
            "pattern_ids",
        )
    )
    patient_id = int(file_name.split("_")[1])
    initial_cluster_x = Initial_Cluster_X[Indices == i]
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Subtrees",
            "Cohort_1",
            file_name,
            "pattern_ids",
            "pattern_id_iter_"
            + str(args.iteration)
            + "_PhenoGraph_k_"
            + str(args.k)
            + ".npy",
        ),
        initial_cluster_x,
    )
    print(f"Saving {file_name}", initial_cluster_x.shape)
