import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT

FILE_NAMES = os.listdir(
    os.path.join(
        PROJECT_ROOT,
        "Output",
        "b_Soft_WL_Kernel",
        "Danenberg",
        "Subtrees",
        "Cohort_1",
    )
)
for i in range(len(FILE_NAMES)):
    file_name = FILE_NAMES[i]
    os.rename(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Subtrees",
            "Cohort_1",
            file_name,
            "pattern_ids",
            "pattern_id_iter_1_PhenoGraph_k_100.npy",
        ),
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "pattern_ids",
            "pattern_id_iter_1_PhenoGraph_k_100.npy",
        ),
    )
    print(f"Renaming {file_name}")
