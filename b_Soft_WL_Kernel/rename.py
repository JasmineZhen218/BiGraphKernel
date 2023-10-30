
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT

FILE_NAMES = os.listdir(os.path.join(PROJECT_ROOT, "Output", "b_Soft_WL_Kernel", "Danenberg", "Cohort_1"))
for i in range(len(FILE_NAMES)):
    file_name = FILE_NAMES[i]
    for iteration in range(1,6):
        os.rename(
            os.path.join(PROJECT_ROOT, "Output", "b_Soft_WL_Kernel", "Danenberg", "Cohort_1",
            file_name,
            "initial_cluster_X"+str(iteration)+ ".npy"),

            os.path.join(PROJECT_ROOT, "Output", "b_Soft_WL_Kernel", "Danenberg", "Cohort_1",
            file_name,
            "initial_cluster_X"+str(iteration)+ "_PhenoGraph_k_30.npy"),
        )
        print(f"Renaming {file_name}", f"initial_cluster_X{iteration}.npy", f"initial_cluster_X{iteration}_PhenoGraph_k_30.npy")