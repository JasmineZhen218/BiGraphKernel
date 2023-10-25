import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
import pandas as pd
from CellGraph import Pos2Adj


FILE_NAMES = os.listdir(os.path.join(PROJECT_ROOT, "Input", "Single-cell", "Danenberg"))
for file_name in FILE_NAMES:
    
    print(f"Processing {file_name}")
    os.makedirs(os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph", "Danenberg", file_name.split('.csv')[0]), exist_ok=True)
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "Input", "Single-cell", "Danenberg", file_name))
    
    Pos = df[["X", "Y"]].values
    Adj = Pos2Adj(Pos)
    np.save(
        os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph", "Danenberg", file_name.split('.csv')[0], "Adj.npy"),
        Adj
    )

    CellType = df["cell_type_id"].values
    np.save(
        os.path.join(PROJECT_ROOT, "Output", "a_Cellular_graph", "Danenberg", file_name.split('.csv')[0], "CellType.npy"),
        CellType
    )


