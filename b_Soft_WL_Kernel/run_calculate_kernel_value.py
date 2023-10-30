import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
import pickle
from SoftWL import get_Gram_matrix
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iteration", type=int, default=2, help="Iteration of neighborhood aggregation"
)
parser.add_argument(
    "--k", type=int, default=100, help="Neighbor of neighborhood in PhenoGraph"
)
args = parser.parse_args()
print(args)

FILE_NAMES = os.listdir(
    os.path.join(
        PROJECT_ROOT, "Output", "b_Soft_WL_Kernel", "Danenberg", "Cohort_1", "Subtrees"
    )
)
assert len(FILE_NAMES) == 467

# Load initial clusters
Pattern_ids = []
for i in range(len(FILE_NAMES)):
    file_name = FILE_NAMES[i]
    pattern_ids = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "pattern_ids",
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
    pattern_ids = np.load(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Subtrees",
            file_name,
            "pattern_ids",
            "pattern_id_iter_"
            + str(args.iteration)
            + "_PhenoGraph_k_"
            + str(args.k)
            + ".npy",
        )
    )
    histogram = np.zeros(num_unique_patterns)
    for j in range(len(np.unique(pattern_ids))):
        histogram[j] = np.sum(pattern_ids == j)
    if patient_id not in Histogram_dict:
        Histogram_dict[patient_id] = histogram
    else:
        Histogram_dict[patient_id] += histogram

Histograms = np.zeros((len(Histogram_dict.keys()), num_unique_patterns))
Patient_ids = []
for i, (patient_id, histogram) in enumerate(Histogram_dict.items()):
    Histograms[i, :] = Histogram_dict[patient_id]
    Patient_ids.append(patient_id)
assert len(Patient_ids) == Histograms.shape[0]
assert Histograms.shape[1] == num_unique_patterns

# Calculate and save Gram matrix
Gram_matrix = get_Gram_matrix(Histograms)
SoftWL_dict = {
    "Patient_id": Patient_ids,
    "Histogram": Histograms,
    "Gram_matrix": Gram_matrix,
}
pickle.dump(
    SoftWL_dict,
    open(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "SoftWL_dict_iter_"
            + str(args.iteration)
            + "_PhenoGraph_k_"
            + str(args.k)
            + ".pkl",
        ),
        "wb",
    ),
)
print("Saving Gram matrix", Gram_matrix.shape)

# Save histohgrams for each patient
for patient_id in Patient_ids:
    histogram = Histogram_dict[patient_id]
    os.makedirs(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Histograms",
            "patient_" + str(patient_id),
        ),
        exist_ok=True,
    )
    np.save(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Histograms",
            "patient_" + str(patient_id),
            "histogram_iter_"
            + str(args.iteration)
            + "PhenoGraph_k_"
            + str(args.k)
            + ".npy",
        ),
        histogram,
    )
    f, ax = plt.subplots(figsize=(5, 3))
    ax.bar(np.arange(len(histogram)), histogram)
    ax.set_xlabel("Pattern ID")
    ax.set_ylabel("Frequency")
    ax.set_ylim([0, np.max(Histograms)])
    f.savefig(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "Histograms",
            "patient_" + str(patient_id),
            "histogram_iter_"
            + str(args.iteration)
            + "PhenoGraph_k_"
            + str(args.k)
            + ".png",
        ),
        dpi=300,
    )
    print("Saving histogram for patient", patient_id, histogram.shape)
