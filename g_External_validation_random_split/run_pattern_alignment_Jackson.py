import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
import argparse
import time
from g_External_validation.Alignment import knn_alignment, centroid_alignment

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_job", type=int, default=-1, help="Number of jobs for parallel computing"
)
parser.add_argument(
    "--iteration", type=int, default=1, help="Iteration of neighborhood aggregation"
)
parser.add_argument(
    "--k", type=int, default=30, help="Number of nearest neighbors for KNN alignment"
)
parser.add_argument(
    "--method",
    type=str,
    default="centroid",
    help="Method for cell type alignment, knn or centroid",
)
parser.add_argument(
    "--node_label",
    type=str,
    default="CellType",
    help="node label: cell_type or cell-category",
)

args = parser.parse_args()
# -------------------------------------------------------------
INPUT_Reference = os.path.join(
    PROJECT_ROOT, "Output", "a_Cellular_graph_random_split", "Danenberg", "Subset_1"
)
INPUT_Query = os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph_random_split", "Jackson")
FILE_NAMES_Reference = os.listdir(INPUT_Reference)
FILE_NAMES_Query = os.listdir(INPUT_Query)

# --------------------------------------------------------------------
X_Reference = []
Label_Reference = []
for i in range(len(FILE_NAMES_Reference)):
    file_name = FILE_NAMES_Reference[i]
    patient_id = int(file_name.split("_")[1])
    X = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel_random_split",
            "Danenberg",
            "Subset_1",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            args.node_label,
            "X" + str(args.iteration) + ".npy",
        )
    )
    Label = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel_random_split",
            "Danenberg",
            "Subset_1",
            "Subtrees",
            file_name,
            "pattern_ids",
            args.node_label,
            "pattern_id_iter_"
            + str(args.iteration)
            + "_PhenoGraph_k_"
            + str(args.k)
            + ".npy",
        )
    )
    X_Reference.append(X)
    Label_Reference.append(Label)
X_Reference = np.concatenate(X_Reference, axis=0)
Label_Reference = np.concatenate(Label_Reference, axis=0)
assert X_Reference.shape[0] == Label_Reference.shape[0]
print(
    f"Concatenated feature shape in Cohort 1 {X_Reference.shape} from {len(FILE_NAMES_Reference)} cellular graphs",
    f"{len(np.unique(Label_Reference))} patterns obtained",
)
# --------------------------------------------------------------------
X_Query = []
Indices_Query = []
for i in range(len(FILE_NAMES_Query)):
    file_name = FILE_NAMES_Query[i]
    patient_id = int(file_name.split("_")[1])
    X = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel_random_split",
            "Jackson",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            args.node_label,
            
            "matched_X" + str(args.iteration) + "_centroid_alignment.npy",
        )
    )
    X_Query.append(X)
    indices = np.zeros(X.shape[0], dtype=int)
    indices[:] = i
    Indices_Query.append(indices)

X_Query = np.concatenate(X_Query, axis=0)
Indices_Query = np.concatenate(Indices_Query, axis=0)
print(
    f"Concatenated feature shape in Cohort 2 {X_Query.shape} from {len(FILE_NAMES_Query)} cellular graphs"
)
# ----------------------------------------------------------------------------------------
# Normalize the data
# X_Reference = (X_Reference - np.mean(X_Reference, axis=0)) / np.std(X_Reference, axis=0)
# X_Query = (X_Query - np.mean(X_Query, axis=0)) / np.std(X_Query, axis=0)
# ----------------------------------------------------------------------------------------
if args.method == "knn":
    # call "export OMP_NUM_THREADS=1" before running to avoid "Too many memory regions" error with Dask
    print("Running KNN alignment")
    start_time = time.time()
    Label_Query_estimated = knn_alignment(
        X_Reference, Label_Reference, X_Query, k=args.k
    )
    print("Finished KNN alignment, taken {} seconds".format(time.time() - start_time))
elif args.method == "centroid":
    print("Running centroid alignment")
    start_time = time.time()
    Label_Query_estimated = centroid_alignment(
        X_Reference, Label_Reference, X_Query, k=args.k
    )
    print(
        "Finished centroid alignment, taken {} seconds".format(time.time() - start_time)
    )
# ----------------------------------------------------------------------------------------
for i in range(len(FILE_NAMES_Query)):
    file_name = FILE_NAMES_Query[i]

    patient_id = int(file_name.split("_")[1])
    pattern_id = Label_Query_estimated[Indices_Query == i]
    if args.method == "knn":
        os.makedirs(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "b_Soft_WL_Kernel_random_split",
                "Jackson",
                "Subtrees",
                file_name,
                "matched_pattern_ids_knn_alignment",
                args.node_label,
            ),
            exist_ok=True,
        )
        np.save(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "b_Soft_WL_Kernel_random_split",
                "Jackson",
                "Subtrees",
                file_name,
                "matched_pattern_ids_knn_alignment",
                args.node_label,
                "pattern_id_iter_"
                + str(args.iteration)
                + "_PhenoGraph_k_"
                + str(args.k)
                + ".npy",
            ),
            pattern_id,
        ),

    elif args.method == "centroid":
        os.makedirs(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "b_Soft_WL_Kernel_random_split",
                "Jackson",
                "Subtrees",
                file_name,
                "matched_pattern_ids_centroid_alignment",
                args.node_label,
            ),
            exist_ok=True,
        )
        np.save(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "b_Soft_WL_Kernel_random_split",
                "Jackson",
                "Subtrees",
                file_name,
                "matched_pattern_ids_centroid_alignment",
                args.node_label,
                "pattern_id_iter_"
                + str(args.iteration)
                + "_PhenoGraph_k_"
                + str(args.k)
                + ".npy",
            ),
            pattern_id,
        ),
    print(f"Saving {file_name}", pattern_id.shape)
