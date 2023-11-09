import os
import sys
from webbrowser import get
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from definitions import get_paired_markers
from utils import PROJECT_ROOT
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--study", type=str, default="Danenberg", help="Dataset")
parser.add_argument("--Subset", type=int, default=1, help="Subset 1 or 2")
args = parser.parse_args()
print(args)

INPUT_ROOT = os.path.join(PROJECT_ROOT, "Input", "Single-cell", args.study)
FILE_NAMES = os.listdir(INPUT_ROOT)
# --------------------------------------------------------------------------
# Read clinical.csv
clinical = pd.read_csv(
    os.path.join(PROJECT_ROOT, "Input", "Clinical", args.study, "clinical.csv")
)
if args.study == "Jackson":
    Patient_ids = [patient_id for patient_id in list(clinical["patient_id"])]
elif args.study == "Danenberg":
    if args.Subset == 1:
        Patient_ids = [
            patient_id
            for patient_id in list(clinical.loc[clinical["Subset_id"] == 1, "patient_id"])
        ]
    elif args.Subset == 2:
        Patient_ids = [
            patient_id
            for patient_id in list(clinical.loc[clinical["Subset_id"] == 2, "patient_id"])
        ]
    else:
        raise ValueError("Invalid cohort number")
else:
    raise ValueError("Invalid study name")
# -------------------------------------------------------------------------
if args.study == "Danenberg":
    OUTPUT_ROOT = os.path.join(
        PROJECT_ROOT,
        "Output",
        "a_Cellular_graph_random_split",
        "Danenberg",
        "Subset_" + str(args.Subset),
    )
elif args.study == "Jackson":
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph_random_split", "Jackson")
else:
    raise ValueError("Invalid study name")
os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True,
)
# --------------------------------------------------------- -----------------
Paired_Markers = get_paired_markers()
if args.study == "Danenberg":
    Markers = [i[0] for i in Paired_Markers]
elif args.study == "Jackson":
    Markers = [i[1] for i in Paired_Markers]
else:
    raise ValueError("Invalid study name")
# ----------------------------------------------------------------------------
for file_name in FILE_NAMES:
    patient_id = int(file_name.split("_")[1])
    if patient_id not in Patient_ids:
        continue
    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            file_name.split(".csv")[0],
        ),
        exist_ok=True,
    )
    df = pd.read_csv(os.path.join(INPUT_ROOT, file_name))
    Protein_expression = df[Markers].values

    # Pos = df[["X", "Y"]].values
    # Adj = Pos2Adj(Pos)
    # CellType = df["cell_type_id"].values

    np.save(
        os.path.join(
            OUTPUT_ROOT,
            file_name.split(".csv")[0],
            "ProteinExpression.npy",
        ),
        Protein_expression,
    )
    print(f"Saving {file_name} to {args.study} Subset {args.Subset}")


# (cell-gnn) zwang@io86:~/Projects/BiGraph/Input/Single-cell/Danenberg$ ls ../../../Output/a_Cellular_graph/Danenberg/Cohort_1 | wc -l
# 467
# (cell-gnn) zwang@io86:~/Projects/BiGraph/Input/Single-cell/Danenberg$ ls ../../../Output/a_Cellular_graph/Danenberg/Cohort_2 | wc -l
# 154
# (cell-gnn) zwang@io86:~/Projects/BiGraph$ ls Output/a_Cellular_graph/Jackson | wc -l
# 270

