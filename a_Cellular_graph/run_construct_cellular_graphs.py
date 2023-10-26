import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
import pandas as pd
from CellGraph import Pos2Adj


FILE_NAMES = os.listdir(os.path.join(PROJECT_ROOT, "Input", "Single-cell", "Danenberg"))
clinical = pd.read_csv(
    os.path.join(PROJECT_ROOT, "Input", "Clinical", "Danenberg", "clinical.csv")
)
cohort_1_patient_ids = [
    patient_id
    for patient_id in list(clinical.loc[clinical["isDiscovery"], "patient_id"])
]
cohort_2_patient_ids = [
    patient_id
    for patient_id in list(clinical.loc[~clinical["isDiscovery"], "patient_id"])
]
print(
    f"{len(cohort_1_patient_ids)} cohort_1 patients, ",
    f"{len(cohort_2_patient_ids)} cohort_2 patients",
)
# 425 cohort_1 patients,  154 cohort_2 patients
os.makedirs(
    os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph", "Danenberg", "Cohort_1"),
    exist_ok=True,
)
os.makedirs(
    os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph", "Danenberg", "Cohort_2"),
    exist_ok=True,
)

for file_name in FILE_NAMES:
    patient_id = int(file_name.split("_")[1])
    df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "Input", "Single-cell", "Danenberg", file_name)
    )
    Pos = df[["X", "Y"]].values
    Adj = Pos2Adj(Pos)
    CellType = df["cell_type_id"].values

    if patient_id in cohort_1_patient_ids:
        os.makedirs(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "a_Cellular_graph",
                "Danenberg",
                "Cohort_1",
                file_name.split(".csv")[0],
            ),
            exist_ok=True,
        )
        np.save(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "a_Cellular_graph",
                "Danenberg",
                "Cohort_1",
                file_name.split(".csv")[0],
                "Adj.npy",
            ),
            Adj,
        )
        np.save(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "a_Cellular_graph",
                "Danenberg",
                "Cohort_1",
                file_name.split(".csv")[0],
                "CellType.npy",
            ),
            CellType,
        )
        print(f"Saving {file_name} to cohort 1")
    elif patient_id in cohort_2_patient_ids:
        os.makedirs(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "a_Cellular_graph",
                "Danenberg",
                "Cohort_2",
                file_name.split(".csv")[0],
            ),
            exist_ok=True,
        )
        np.save(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "a_Cellular_graph",
                "Danenberg",
                "Cohort_2",
                file_name.split(".csv")[0],
                "Adj.npy",
            ),
            Adj,
        )
        np.save(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "a_Cellular_graph",
                "Danenberg",
                "Cohort_2",
                file_name.split(".csv")[0],
                "CellType.npy",
            ),
            CellType,
        )
        print(f"Saving {file_name} to cohort 2")
    else:
        print(f"Patient {patient_id} not in clinical.csv")
        break

# (cell-gnn) zwang@io86:~/Projects/BiGraph/Input/Single-cell/Danenberg$ ls ../../../Output/a_Cellular_graph/Danenberg/Cohort_1 | wc -l
# 467
# (cell-gnn) zwang@io86:~/Projects/BiGraph/Input/Single-cell/Danenberg$ ls ../../../Output/a_Cellular_graph/Danenberg/Cohort_2 | wc -l
# 154