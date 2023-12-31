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
    default="TMECellType",
    help="node label: cell_type or cell-category",
)


args = parser.parse_args()
# -------------------------------------------------------------
INPUT_Reference = os.path.join(
    PROJECT_ROOT, "Output", "a_Cellular_graph_random_split", "Danenberg", "Subset_1"
)
INPUT_Query = os.path.join(
    PROJECT_ROOT, "Output", "a_Cellular_graph_random_split", "Jackson"
)
OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT, "Output", "a_Cellular_graph_random_split", "Jackson"
)
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
            INPUT_Reference,
            file_name,
            "ProteinExpression.npy",
        )
    )
    Label = np.load(
        os.path.join(
            INPUT_Reference,
            file_name,
            args.node_label + ".npy",
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
            INPUT_Query,
            file_name,
            "ProteinExpression.npy",
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
X_Reference = (X_Reference - np.mean(X_Reference, axis=0)) / np.std(X_Reference, axis=0)
X_Query = (X_Query - np.mean(X_Query, axis=0)) / np.std(X_Query, axis=0)
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
    os.makedirs(
        os.path.join(OUTPUT_ROOT, file_name, args.method + "_alignment"),
        exist_ok=True,
    )
    patient_id = int(file_name.split("_")[1])
    cell_type = Label_Query_estimated[Indices_Query == i]
    if args.method == "knn":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "matched_"
                + args.node_label
                + "_knn_k_"
                + str(args.k)
                + "_alignment.npy",
            ),
            cell_type,
        )
    elif args.method == "centroid":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "matched_" + args.node_label + "_centroid_alignment.npy",
            ),
            cell_type,
        )
    print(f"Saving {file_name}", cell_type.shape)
