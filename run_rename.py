import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import shutil

FILE_NAMES = os.listdir(
    os.path.join(
        PROJECT_ROOT,
        "Output",
        "a_Cellular_graph",
        "Jackson",
    )
)
for i in range(len(FILE_NAMES)):
    file_name = FILE_NAMES[i]
    shutil.rmtree(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "a_Cellular_graph",
            "Jackson",
            file_name,
            'centroid_alignment'
        )
    )
    print(f"Removing {file_name}/centroid_alignment")
    # os.makedirs(
    #     os.path.join(
    #         PROJECT_ROOT,
    #         "Output",
    #         "b_Soft_WL_Kernel",
    #         "Danenberg",
    #         "Cohort_2",
    #         "Subtrees",
    #         file_name,
    #         "neighborhood_aggregation",
    #     ),
    #     exist_ok=True,
    # )
    # for iteration in range(0,1):
    #     os.rename(
    #         os.path.join(
    #             PROJECT_ROOT,
    #             "Output",
    #             "b_Soft_WL_Kernel",
    #             "Danenberg",
    #             "Cohort_2",
    #             file_name,
    #             "X" + str(iteration) + ".npy",
    #         ),
    #         os.path.join(
    #             PROJECT_ROOT,
    #             "Output",
    #             "b_Soft_WL_Kernel",
    #             "Danenberg",
    #             "Cohort_2",
    #             "Subtrees",
    #             file_name,
    #             "neighborhood_aggregation",
    #             "X" + str(iteration) + ".npy",
    #         ),
    #     )
    #     print(f"Renaming {file_name}")
