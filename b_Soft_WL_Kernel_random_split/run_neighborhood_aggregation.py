import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
from SoftWL import neighborhood_aggregation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--study", type=str, default="Danenberg", help="Dataset")
parser.add_argument("--Subset", type=int, default=1, help="Subset 1 or 2")
parser.add_argument(
    "--node_label",
    type=str,
    default="CellType",
    help="node label: cell_type or cell-category",
)
args = parser.parse_args()
print(args)

if args.study == "Jackson":
    INPUT_ROOT = os.path.join(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "a_Cellular_graph_random_split",
            "Jackson",
        )
    )
    FILE_NAMES = os.listdir(INPUT_ROOT)
    OUTPUT_ROOT = os.path.join(
        PROJECT_ROOT,
        "Output",
        "b_Soft_WL_Kernel_random_split",
        "Jackson",
        "Subtrees",
    )
elif args.study == "Danenberg":
    INPUT_ROOT = os.path.join(
        PROJECT_ROOT,
        "Output",
        "a_Cellular_graph_random_split",
        "Danenberg",
        "Subset_" + str(args.Subset),
    )
    FILE_NAMES = os.listdir(INPUT_ROOT)
    OUTPUT_ROOT = os.path.join(
        PROJECT_ROOT,
        "Output",
        "b_Soft_WL_Kernel_random_split",
        "Danenberg",
        "Subset_" + str(args.Subset),
        "Subtrees",
    )
else:
    raise ValueError("Invalid study name")


for file_name in FILE_NAMES:
    print(f"Processing {file_name}")
    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            file_name,
            "neighborhood_aggregation",
            args.node_label,
        ),
        exist_ok=True,
    )
    Adj = np.load(
        os.path.join(
            INPUT_ROOT,
            file_name,
            "Adj.npy",
        )
    )
    if args.study == "Jackson":
        CellType = np.load(
            os.path.join(
                PROJECT_ROOT,
                INPUT_ROOT,
                file_name,
                "matched_" + args.node_label + "_centroid_alignment.npy",
            )
        ).reshape(-1)
    elif args.study == "Danenberg":
        CellType = np.load(
            os.path.join(
                PROJECT_ROOT,
                INPUT_ROOT,
                file_name,
                args.node_label + ".npy",
            )
        ).reshape(-1)
    # --------------------------------------------------------------------------
    if args.node_label == "CellType":
        X0 = np.zeros((Adj.shape[0], 32))
        for i in range(32):
            X0[CellType == i, i] = 1
    elif args.node_label == "CellCategory":
        X0 = np.zeros((Adj.shape[0], 3))
        for i in range(3):
            X0[CellType == i, i] = 1
    elif args.node_label == "TMECellType":
        print(np.unique(CellType))
        X0 = np.zeros((Adj.shape[0], 17))
        for i in range(17):
            X0[CellType == i, i] = 1
    if args.study == "Jackson":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "matched_X0_centroid_alignment.npy",
            ),
            X0,
        )
    elif args.study == "Danenberg":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "X0.npy",
            ),
            X0,
        )
    print(f"Saved X0.npy")
    # --------------------------------------------------------------------------
    X = neighborhood_aggregation(X0, Adj, 1)
    if args.study == "Jackson":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "matched_X1_centroid_alignment.npy",
            ),
            X,
        )
    elif args.study == "Danenberg":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "X1.npy",
            ),
            X,
        )
    print(f"Saved X1.npy")
    # --------------------------------------------------------------------------
    X = neighborhood_aggregation(X, Adj, 1)
    if args.study == "Jackson":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "matched_X2_centroid_alignment.npy",
            ),
            X,
        )
    elif args.study == "Danenberg":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "X2.npy",
            ),
            X,
        )
    print(f"Saved X2.npy")
    # --------------------------------------------------------------------------
    X = neighborhood_aggregation(X, Adj, 1)
    if args.study == "Jackson":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "matched_X3_centroid_alignment.npy",
            ),
            X,
        )
    elif args.study == "Danenberg":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "X3.npy",
            ),
            X,
        )
    print(f"Saved X3.npy")
    # --------------------------------------------------------------------------
    X = neighborhood_aggregation(X, Adj, 1)
    if args.study == "Jackson":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "matched_X4_centroid_alignment.npy",
            ),
            X,
        )
    elif args.study == "Danenberg":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "X4.npy",
            ),
            X,
        )
    print(f"Saved X4.npy")
    # --------------------------------------------------------------------------
    X = neighborhood_aggregation(X, Adj, 1)
    if args.study == "Jackson":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "matched_X5_centroid_alignment.npy",
            ),
            X,
        )
    elif args.study == "Danenberg":
        np.save(
            os.path.join(
                OUTPUT_ROOT,
                file_name,
                "neighborhood_aggregation",
                args.node_label,
                "X5.npy",
            ),
            X,
        )
    print(f"Saved X5.npy")
# --------------------------------------------------------------------------

# print("Cohort 2")
# FILE_NAMES = os.listdir(
#     os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph", "Danenberg", "Cohort_2")
# )
# for file_name in FILE_NAMES:
#     print(f"Processing {file_name}")
#     os.makedirs(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel",
#             "Danenberg",
#             "Cohort_2",
#             "Subtrees",
#             file_name,
#             "neighborhood_aggregation",
#         ),
#         exist_ok=True,
#     )
#     Adj = np.load(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "a_Cellular_graph",
#             "Danenberg",
#             "Cohort_2",
#             file_name,
#             "Adj.npy",
#         )
#     )
#     CellType = np.load(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "a_Cellular_graph",
#             "Danenberg",
#             "Cohort_2",
#             file_name,
#             "CellType.npy",
#         )
#     )
#     X0 = np.zeros((Adj.shape[0], 32))
#     for i in range(32):
#         X0[CellType == i, i] = 1
#     np.save(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel",
#             "Danenberg",
#             "Cohort_2",
#             "Subtrees",
#             file_name,
#             "neighborhood_aggregation",
#             "X0.npy",
#         ),
#         X0,
#     )
#     print(f"Saved X0.npy")
#     X = neighborhood_aggregation(X0, Adj, 1)
#     np.save(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel",
#             "Danenberg",
#             "Cohort_2",
#             "Subtrees",
#             file_name,
#             "neighborhood_aggregation",
#             "X1.npy",
#         ),
#         X,
#     )
#     print(f"Saved X1.npy")
#     X = neighborhood_aggregation(X, Adj, 1)
#     np.save(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel",
#             "Danenberg",
#             "Cohort_2",
#             "Subtrees",
#             file_name,
#             "neighborhood_aggregation",
#             "X2.npy",
#         ),
#         X,
#     )
#     print(f"Saved X2.npy")
#     X = neighborhood_aggregation(X, Adj, 1)
#     np.save(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel",
#             "Danenberg",
#             "Cohort_2",
#             "Subtrees",
#             file_name,
#             "neighborhood_aggregation",
#             "X3.npy",
#         ),
#         X,
#     )
#     print(f"Saved X3.npy")
#     X = neighborhood_aggregation(X, Adj, 1)
#     np.save(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel",
#             "Danenberg",
#             "Cohort_2",
#             "Subtrees",
#             file_name,
#             "neighborhood_aggregation",
#             "X4.npy",
#         ),
#         X,
#     )
#     print(f"Saved X4.npy")
#     X = neighborhood_aggregation(X, Adj, 1)
#     np.save(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel",
#             "Danenberg",
#             "Cohort_2",
#             "Subtrees",
#             file_name,
#             "neighborhood_aggregation",
#             "X5.npy",
#         ),
#         X,
#     )
#     print(f"Saved X5.npy")


# (cell-gnn) zwang@io86:~/Projects/BiGraph/Output/b_Soft_WL_Kernel/Danenberg$ ls Cohort_1 | wc -l
# 467
# (cell-gnn) zwang@io86:~/Projects/BiGraph/Output/b_Soft_WL_Kernel/Danenberg$ ls Cohort_2 | wc -l
# 154
