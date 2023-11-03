import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
import argparse
from KNN_alignment import knn_alignment

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_job", type=int, default=-1, help="Number of jobs for parallel computing"
)
parser.add_argument(
    "--iteration", type=int, default=2, help="Iteration of neighborhood aggregation"
)
parser.add_argument(
    "--k", type=int, default=30, help="Neighbor of neighborhood in PhenoGraph"
)
args = parser.parse_args()


FILE_NAMES_Cohort1 = os.listdir(
    os.path.join(
        PROJECT_ROOT, "Output", "b_Soft_WL_Kernel", "Danenberg", "Cohort_1", "Subtrees"
    )
)
FILE_NAMES_Cohort2 = os.listdir(
    os.path.join(
        PROJECT_ROOT, "Output", "b_Soft_WL_Kernel", "Danenberg", "Cohort_2", "Subtrees"
    )
)
assert len(FILE_NAMES_Cohort1) == 467
# Concatenate all subtrees from all cellular graphs
X_Cohort1 = []
Label_Cohort1 = []
for i in range(len(FILE_NAMES_Cohort1)):
    file_name = FILE_NAMES_Cohort1[i]
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
    pattern_id = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "pattern_ids",
            "pattern_id_iter_"
            + str(args.iteration)
            + "_PhenoGraph_k_"
            + str(args.k)
            + ".npy",
        )
    )
    X_Cohort1.append(X)
    Label_Cohort1.append(pattern_id)
X_Cohort1 = np.concatenate(X_Cohort1, axis=0)
Label_Cohort1 = np.concatenate(Label_Cohort1, axis=0)
assert X_Cohort1.shape[0] == Label_Cohort1.shape[0]
print(
    f"Concatenated feature shape in Cohort 1 {X_Cohort1.shape} from {len(FILE_NAMES_Cohort1)} cellular graphs",
    f"{len(np.unique(Label_Cohort1))} patterns obtained",
)

X_Cohort2 = []
Indices_Cohort2 = []
for i in range(len(FILE_NAMES_Cohort2)):
    file_name = FILE_NAMES_Cohort2[i]
    patient_id = int(file_name.split("_")[1])
    X = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X" + str(args.iteration) + ".npy",
        )
    )
    X_Cohort2.append(X)
    indices = np.zeros(X.shape[0], dtype=int)
    indices[:] = i
    Indices_Cohort2.append(indices)

X_Cohort2 = np.concatenate(X_Cohort2, axis=0)
Indices_Cohort2 = np.concatenate(Indices_Cohort2, axis=0)

print(
    f"Concatenated feature shape in Cohort 2 {X_Cohort2.shape} from {len(FILE_NAMES_Cohort2)} cellular graphs"
)

# call "export OMP_NUM_THREADS=1" before running to avoid "Too many memory regions" error with Dask
Label_Cohort2_estimated = knn_alignment(
    X_Cohort1, Label_Cohort1, X_Cohort2, k=args.k
)


# Save matched pattern_ids
for i in range(len(FILE_NAMES_Cohort2)):
    file_name = FILE_NAMES_Cohort2[i]
    os.makedirs(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "matched_pattern_ids",
        ),
        exist_ok=True,
    )
    patient_id = int(file_name.split("_")[1])
    pattern_id = Label_Cohort2_estimated[Indices_Cohort2 == i]
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "matched_pattern_ids",
            "pattern_id_iter_"
            + str(args.iteration)
            + "_PhenoGraph_k_"
            + str(args.k)
            + ".npy",
        ),
        pattern_id,
    )
    print(f"Saving {file_name}", pattern_id.shape)
