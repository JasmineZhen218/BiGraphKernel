import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import shutil

FILE_NAMES = os.listdir(
    os.path.join(
        PROJECT_ROOT,
        "Output",
        "b_Soft_WL_Kernel_random_split",
        "Jackson",
        "Subtrees",
    )
)
for iteration in range(1, 6):
    for k in [30, 100, 200, 500]:
        os.rename(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "b_Soft_WL_Kernel_random_split",
                "Danenberg",
                "Subset_1",
                "SoftWL_dict_iter_"
                + str(iteration)
                + "_PhenoGraph_k_"
                + str(k)
                + ".pkl",
            ),
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "b_Soft_WL_Kernel_random_split",
                   "Danenberg",
                "Subset_1",
                "SoftWL_dict_iter_"
                + str(iteration)
                + "_PhenoGraph_k_"
                + str(k)
                + "_CellType.pkl",
            ),
        )
        # print(f"Renaming {file_name}")


# for i in range(len(FILE_NAMES)):
#     file_name = FILE_NAMES[i]
#     # shutil.rmtree(
#     #     os.path.join(
#     #         PROJECT_ROOT,
#     #         "Output",
#     #         "b_Soft_WL_Kernel_random_split",
#     #         "Danenberg",
#     #         "Subset_1",
#     #         "Subtrees",
#     #         file_name,
#     #         "centroid_alignment",
#     #     )
#     # )
#     # print(f"Removing {file_name}/centroid_alignment")
#     os.makedirs(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel_random_split",
#             "Jackson",
#             "Subtrees",
#             file_name,
#             "matched_pattern_ids_centroid_alignment",
#             "CellType",
#         ),
#         exist_ok=True,
#     )
#     for iteration in range(1, 6):
#         for k in [30, 100, 200, 500]:
#             os.rename(
#                 os.path.join(
#                     PROJECT_ROOT,
#                     "Output",
#                     "b_Soft_WL_Kernel_random_split",
#                     "Jackson",
#                     "Subtrees",
#                     file_name,
#                     "matched_pattern_ids_centroid_alignment",
#                     "pattern_id_iter_"
#                     + str(iteration)
#                     + "_PhenoGraph_k_"
#                     + str(k)
#                     + ".npy",
#                 ),
#                 os.path.join(
#                     PROJECT_ROOT,
#                     "Output",
#                     "b_Soft_WL_Kernel_random_split",
#                     "Jackson",
#                     "Subtrees",
#                     file_name,
#                     "matched_pattern_ids_centroid_alignment",
#                     "CellType",
#                     "pattern_id_iter_"
#                     + str(iteration)
#                     + "_PhenoGraph_k_"
#                     + str(k)
#                     + ".npy",
#                 ),
#             )
#             print(f"Renaming {file_name}")
