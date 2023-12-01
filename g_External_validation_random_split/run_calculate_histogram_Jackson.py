import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iteration", type=int, default=0, help="Iteration of neighborhood aggregation"
)
parser.add_argument(
    "--k", type=int, default=100, help="Neighbor of neighborhood in PhenoGraph"
)
parser.add_argument(
    "--method", type=str, default="centroid", help="Method for cell type alignment"
)
parser.add_argument(
    "--node_label",
    type=str,
    default="CellCategory",
    help="node label: cell_type or cell-category",
)
args = parser.parse_args()
print(args)

FILE_NAMES = os.listdir(
    os.path.join(
        PROJECT_ROOT, "Output", "b_Soft_WL_Kernel_random_split", "Jackson", "Subtrees"
    )
)

# Load initial clusters
Pattern_ids = []
for i in range(len(FILE_NAMES)):
    file_name = FILE_NAMES[i]
    if args.iteration == 0:
        pattern_ids = np.load(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "a_Cellular_graph_random_split",
                "Jackson",
                file_name,
                "matched_" + args.node_label + "_centroid_alignment.npy",
            )
        )
    else:
        pattern_ids = np.load(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "b_Soft_WL_Kernel_random_split",
                "Jackson",
                "Subtrees",
                file_name,
                "matched_pattern_ids_centroid_alignment",
                args.node_label,
                "pattern_id_iter_"
                + str(args.iteration)
                + "_PhenoGraph_k_"
                + str(args.k)
                + ".npy",
            )
        )
    Pattern_ids.append(pattern_ids)
    print(f"Loading {file_name}", pattern_ids.shape)
assert len(Pattern_ids) == len(FILE_NAMES)
Pattern_ids = np.concatenate(Pattern_ids, axis=0)
num_unique_patterns = len(np.unique(Pattern_ids))
print("Number of unique patterns", num_unique_patterns)


# calculate histogram for each patient
Histogram_dict = {}
for i in range(len(FILE_NAMES)):
    file_name = FILE_NAMES[i]
    patient_id = int(file_name.split("_")[1])
    if args.iteration == 0:
        pattern_ids = np.load(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "a_Cellular_graph_random_split",
                "Jackson",
                file_name,
                "matched_" + args.node_label + "_centroid_alignment.npy",
            )
        )
    else:
        pattern_ids = np.load(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "b_Soft_WL_Kernel_random_split",
                "Jackson",
                "Subtrees",
                file_name,
                "matched_pattern_ids_centroid_alignment",
                args.node_label,
                "pattern_id_iter_"
                + str(args.iteration)
                + "_PhenoGraph_k_"
                + str(args.k)
                + ".npy",
            )
        )
    histogram = np.zeros(num_unique_patterns)
    for j in range(num_unique_patterns):
        histogram[j] = np.sum(pattern_ids == j)
    assert np.sum(histogram) != 0
    if patient_id not in Histogram_dict:
        Histogram_dict[patient_id] = histogram
    else:
        Histogram_dict[patient_id] += histogram

Histograms = np.zeros((len(Histogram_dict.keys()), num_unique_patterns))
Patient_ids = []
for i, (patient_id, histogram) in enumerate(Histogram_dict.items()):
    assert np.sum(histogram) != 0
    Histograms[i, :] = Histogram_dict[patient_id]
    Patient_ids.append(patient_id)
assert len(Patient_ids) == Histograms.shape[0]
assert Histograms.shape[1] == num_unique_patterns

# Calculate and save Gram matrix
SoftWL_dict = {
    "Patient_id": Patient_ids,
    "Histogram": Histograms,
}
if args.iteration == 0:
    pickle.dump(
        SoftWL_dict,
        open(
            os.path.join(
                PROJECT_ROOT,
                "Output",
                "b_Soft_WL_Kernel_random_split",
                "Jackson",
                "Matched_SoftWL_dict_iter_"
                + str(args.iteration)
                + "_"
                + args.node_label
                + "_centroid_alignment.pkl",
            ),
            "wb",
        ),
    )

pickle.dump(
    SoftWL_dict,
    open(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel_random_split",
            "Jackson",
            "Matched_SoftWL_dict_iter_"
            + str(args.iteration)
            + "_PhenoGraph_k_"
            + str(args.k)
            + "_"
            + args.node_label
            + "_centroid_alignment.pkl",
        ),
        "wb",
    ),
)

# Save histohgrams for each patient
# for patient_id in Patient_ids:
#     histogram = Histogram_dict[patient_id]
#     os.makedirs(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel_random_split",
#             "Jackson",
#             "Matched_Histograms_centroid_alignment",
#             "patient_" + str(patient_id),
#         ),
#         exist_ok=True,
#     )
#     np.save(
#         os.path.join(
#             PROJECT_ROOT,
#             "Output",
#             "b_Soft_WL_Kernel_random_split",
#             "Jackson",
#             "Matched_Histograms_centroid_alignment",
#             "patient_" + str(patient_id),
#             "histogram_iter_"
#             + str(args.iteration)
#             + "_PhenoGraph_k_"
#             + str(args.k)
#             + ".npy",
#         ),
#         histogram,
#     )
#     print("Saving histogram for patient", patient_id, histogram.shape)
