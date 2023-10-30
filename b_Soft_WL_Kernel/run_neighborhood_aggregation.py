import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
from SoftWL import neighborhood_aggregation

print("Cohort 1")
FILE_NAMES = os.listdir(
    os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph", "Danenberg", "Cohort_1")
)
for file_name in FILE_NAMES:
    print(f"Processing {file_name}")
    os.makedirs(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
        ),
        exist_ok=True,
    )
    Adj = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "a_Cellular_graph",
            "Danenberg",
            "Cohort_1",
            file_name,
            "Adj.npy",
        )
    )
    CellType = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "a_Cellular_graph",
            "Danenberg",
            "Cohort_1",
            file_name,
            "CellType.npy",
        )
    )
    X0 = np.zeros((Adj.shape[0], 32))
    for i in range(32):
        X0[CellType == i, i] = 1
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X0.npy",
        ),
        X0,
    )
    print(f"Saved X0.npy")
    X = neighborhood_aggregation(X0, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X1.npy",
        ),
        X,
    )
    print(f"Saved X1.npy")
    X = neighborhood_aggregation(X, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X2.npy",
        ),
        X,
    )
    print(f"Saved X2.npy")
    X = neighborhood_aggregation(X, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X3.npy",
        ),
        X,
    )
    print(f"Saved X3.npy")
    X = neighborhood_aggregation(X, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X4.npy",
        ),
        X,
    )
    print(f"Saved X4.npy")
    X = neighborhood_aggregation(X, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X5.npy",
        ),
        X,
    )
    print(f"Saved X5.npy")


print("Cohort 2")
FILE_NAMES = os.listdir(
    os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph", "Danenberg", "Cohort_2")
)
for file_name in FILE_NAMES:
    print(f"Processing {file_name}")
    os.makedirs(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
        ),
        exist_ok=True,
    )
    Adj = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "a_Cellular_graph",
            "Danenberg",
            "Cohort_2",
            file_name,
            "Adj.npy",
        )
    )
    CellType = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "a_Cellular_graph",
            "Danenberg",
            "Cohort_2",
            file_name,
            "CellType.npy",
        )
    )
    X0 = np.zeros((Adj.shape[0], 32))
    for i in range(32):
        X0[CellType == i, i] = 1
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X0.npy",
        ),
        X0,
    )
    print(f"Saved X0.npy")
    X = neighborhood_aggregation(X0, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X1.npy",
        ),
        X,
    )
    print(f"Saved X1.npy")
    X = neighborhood_aggregation(X, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X2.npy",
        ),
        X,
    )
    print(f"Saved X2.npy")
    X = neighborhood_aggregation(X, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X3.npy",
        ),
        X,
    )
    print(f"Saved X3.npy")
    X = neighborhood_aggregation(X, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X4.npy",
        ),
        X,
    )
    print(f"Saved X4.npy")
    X = neighborhood_aggregation(X, Adj, 1)
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_2",
            "Subtrees",
            file_name,
            "neighborhood_aggregation",
            "X5.npy",
        ),
        X,
    )
    print(f"Saved X5.npy")


# (cell-gnn) zwang@io86:~/Projects/BiGraph/Output/b_Soft_WL_Kernel/Danenberg$ ls Cohort_1 | wc -l
# 467
# (cell-gnn) zwang@io86:~/Projects/BiGraph/Output/b_Soft_WL_Kernel/Danenberg$ ls Cohort_2 | wc -l
# 154
