from logging import raiseExceptions
import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    PROJECT_ROOT,
    process_Danenberg_clinical_data,
    process_Jackson_clinical_data,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import networkx as nx
from matplotlib.lines import Line2D
import pandas as pd
from c_Population_graph.population_graph import construct_PopulationGraph
from d_Patient_subgroups.community_detection import detect_communities
from survival_analysis import calculate_hazard_ratio
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
import warnings

warnings.filterwarnings("ignore")

node_label = "CellType"
size_smallest_cluster = 10
survival_type = "Disease-specific"
PopulationGraph_type = "complete_graph"

for iteration in range(1, 6):
    for PhenoGraph_k in [30, 100, 200, 500]:
        # for knn_k in [10,20,30,40,50]:
        #     para_dict = {"knn_k": knn_k}
            SoftWL_dict = pickle.load(
                open(
                    os.path.join(
                        PROJECT_ROOT,
                        "Output",
                        "b_Soft_WL_Kernel_random_split",
                        "Danenberg",
                        "Subset_1",
                        "SoftWL_dict_iter_"
                        + str(iteration)
                        + "_PhenoGraph_k_"
                        + str(PhenoGraph_k)
                        + "_"
                        + node_label
                        + ".pkl",
                    ),
                    "rb",
                ),
            )
            Patient_IDs = SoftWL_dict["Patient_id"]
            Gram_matrix = SoftWL_dict["Gram_matrix"]
            Histograms = SoftWL_dict["Histogram"]
            Histograms = Histograms / np.sum(Histograms, axis=1, keepdims=True)
            clinical = pd.read_csv(
                os.path.join(
                    PROJECT_ROOT,
                    "Input",
                    "Clinical",
                    "Danenberg",
                    "clinical.csv",
                )
            )
            G_population = construct_PopulationGraph(
                Gram_matrix,
                PopulationGraph_type,
                # para_dict=para_dict,
            )
            Community_ids = detect_communities(
                G_population, size_smallest_cluster, resolution=1
            )
            if survival_type == "Overall":
                Length = [
                    clinical.loc[
                        clinical["patient_id"] == i, "Overall Survival (Months)"
                    ].values[0]
                    for i in Patient_IDs
                ]
                Status = [
                    clinical.loc[
                        clinical["patient_id"] == i, "Overall Survival Status"
                    ].values[0]
                    for i in Patient_IDs
                ]
            elif survival_type == "Relpase-free":
                Length = [
                    clinical.loc[
                        clinical["patient_id"] == i, "Relapse Free Status (Months)"
                    ].values[0]
                    for i in Patient_IDs
                ]
                Status = [
                    clinical.loc[
                        clinical["patient_id"] == i, "Relapse Free Status"
                    ].values[0]
                    for i in Patient_IDs
                ]
            elif survival_type == "Disease-specific":
                Length = [
                    clinical.loc[
                        clinical["patient_id"] == i,
                        "Disease-specific Survival (Months)",
                    ].values[0]
                    for i in Patient_IDs
                ]
                Status = [
                    clinical.loc[
                        clinical["patient_id"] == i, "Disease-specific Survival Status"
                    ].values[0]
                    for i in Patient_IDs
                ]
            DF = pd.DataFrame(
                {"Length": Length, "Status": Status, "Community_ids": Community_ids}
            ).dropna()

            Length_ = np.array(DF["Length"])
            Status_ = np.array(DF["Status"])
            Community_ids_ = np.array(DF["Community_ids"])
            Patient_IDs_ = np.array(Patient_IDs)[DF.index]
            Histograms_ = Histograms[DF.index.values, :]

            HR = calculate_hazard_ratio(
                Length_,
                Status_,
                Community_ids_,
            )
            num = 0
            for i in range(len(HR)):
                if HR[i]["p"] < 0.05/len(HR) :
                    num += 1
            if num > 0:
                print(
                    "---------------------------------------------\niteration: {}, PhenoGraph_k: {}".format(
                        iteration, PhenoGraph_k, 
                    )
                )

                HR = sorted(HR, key=lambda x: x["hr"], reverse=True)
                Subgroup_ids = np.zeros_like(Community_ids)
                Subgroup_ids_ = np.zeros_like(Community_ids_)
                for i in range(len(HR)):
                    Subgroup_ids[Community_ids == HR[i]["community_id"]] = i + 1
                    Subgroup_ids_[Community_ids_ == HR[i]["community_id"]] = i + 1
                    HR[i]["subgroup_id"] = i + 1

                # Overpresented_patterns = {}
                # Candiates_overpresented_in_subgroup = []
                # from scipy import stats

                # for hr_dict in HR:
                #     subgroup_id = hr_dict["subgroup_id"]
                #     hr = hr_dict["hr"]
                #     p = hr_dict["p"]
                #     if p < 0.05 :
                #         Overpresented_patterns[subgroup_id] = []
                #         Histogram_intra_group = Histograms_[
                #             Subgroup_ids_ == subgroup_id
                #         ]
                #         for other_subgroup_id in [
                #             i for i in range(1, len(HR) + 1) if i != subgroup_id
                #         ]:
                #             Histogram_other_group = Histograms_[
                #                 Subgroup_ids_ == other_subgroup_id
                #             ]
                #             candidates = []
                #             for i in range(Histograms_.shape[1]):
                #                 rvsi = Histogram_intra_group[:, i]
                #                 rvso = Histogram_other_group[:, i]
                #                 result = stats.mannwhitneyu(rvsi, rvso)
                #                 if (result.pvalue < 0.05 ) and (
                #                     np.median(rvsi) > np.median(rvso)
                #                 ):
                #                     candidates.append(i)
                #             Overpresented_patterns[subgroup_id].append(set(candidates))
                #         Overpresented_patterns[subgroup_id] = list(
                #             set.intersection(*Overpresented_patterns[subgroup_id])
                #         )
                # for subgroup_id, Candidates in Overpresented_patterns.items():
                #     print(
                #         "subgroup_id: {} (N = {}), hr = {:.4f}, p = {:.4f}".format(
                #             subgroup_id, np.sum(Subgroup_ids_ == subgroup_id), HR[subgroup_id - 1]["hr"], HR[subgroup_id - 1]["p"]
                #         )
                #     )
                #     print(Candidates)

                # Num_patterns_to_test = 0
                # for subgroup_id, Candidates in Overpresented_patterns.items():
                #     Num_patterns_to_test += len(Candidates)

                # # Subset 2
                # SoftWL_dict_cohort2 = pickle.load(
                #     open(
                #         os.path.join(
                #             PROJECT_ROOT,
                #             "Output",
                #             "b_Soft_WL_Kernel_random_split",
                #             "Danenberg",
                #             "Subset_2",
                #             "Matched_SoftWL_dict_iter_"
                #             + str(iteration)
                #             + "_PhenoGraph_k_"
                #             + str(PhenoGraph_k)
                #             + "_"
                #             + node_label
                #             + "_centroid_alignment.pkl",
                #         ),
                #         "rb",
                #     ),
                # )
                # Patient_IDs_cohort2 = SoftWL_dict_cohort2["Patient_id"]
                # Histograms_cohort2 = SoftWL_dict_cohort2["Histogram"]
                # Histograms_cohort2 = Histograms_cohort2 / np.sum(
                #     Histograms_cohort2, axis=1, keepdims=True
                # )

                # if survival_type == "Overall":
                #     Length_cohort2 = [
                #         clinical.loc[
                #             clinical["patient_id"] == i, "Overall Survival (Months)"
                #         ].values[0]
                #         for i in Patient_IDs_cohort2
                #     ]
                #     Status_cohort2 = [
                #         clinical.loc[
                #             clinical["patient_id"] == i, "Overall Survival Status"
                #         ].values[0]
                #         for i in Patient_IDs_cohort2
                #     ]
                # elif survival_type == "Relpase-free":
                #     Length_cohort2 = [
                #         clinical.loc[
                #             clinical["patient_id"] == i, "Relapse Free Status (Months)"
                #         ].values[0]
                #         for i in Patient_IDs_cohort2
                #     ]
                #     Status_cohort2 = [
                #         clinical.loc[
                #             clinical["patient_id"] == i, "Relapse Free Status"
                #         ].values[0]
                #         for i in Patient_IDs_cohort2
                #     ]
                # elif survival_type == "Disease-specific":
                #     Length_cohort2 = [
                #         clinical.loc[
                #             clinical["patient_id"] == i,
                #             "Disease-specific Survival (Months)",
                #         ].values[0]
                #         for i in Patient_IDs_cohort2
                #     ]
                #     Status_cohort2 = [
                #         clinical.loc[
                #             clinical["patient_id"] == i,
                #             "Disease-specific Survival Status",
                #         ].values[0]
                #         for i in Patient_IDs_cohort2
                #     ]
                # DF_cohort2 = pd.DataFrame(
                #     {"Length": Length_cohort2, "Status": Status_cohort2}
                # ).dropna()
                # Length_cohort2_ = np.array(DF_cohort2["Length"])
                # Status_cohort2_ = np.array(DF_cohort2["Status"])
                # Histograms_cohort2_ = Histograms_cohort2[DF_cohort2.index.values, :]
                # Patient_IDs_cohort2_ = np.array(Patient_IDs_cohort2)[
                #     DF_cohort2.index.values
                # ]

                # DF = pd.DataFrame(
                #     {
                #         "length": Length_cohort2_,
                #         "status": Status_cohort2_,
                #     }
                # )

                # for subgroup_id, Candidates in Overpresented_patterns.items():
                #     for pattern_id in Candidates:
                #         histogram = Histograms_cohort2_[:, pattern_id]
                #         DF["pattern_" + str(pattern_id + 1)] = histogram > 0
                #         if np.sum(DF["pattern_" + str(pattern_id + 1)]) == 0:
                #             continue
                #         try:
                #             cph = CoxPHFitter()
                #             cph.fit(
                #                 DF,
                #                 duration_col="length",
                #                 event_col="status",
                #                 formula="pattern_" + str(pattern_id + 1),
                #             )
                #             hr = cph.hazard_ratios_["pattern_" + str(pattern_id + 1)]
                #             p = cph.summary["p"]["pattern_" + str(pattern_id + 1)]
                #             if p < 0.05 / Num_patterns_to_test:
                #                 print(
                #                     "[Subset 2, Existence] Pattern: {}, subgroup_id: {}, HR: {}, p: {}".format(
                #                         pattern_id + 1, subgroup_id, hr, p
                #                     )
                #                 )
                #         except:
                #             continue

                # SoftWL_dict_jackson = pickle.load(
                #     open(
                #         os.path.join(
                #             PROJECT_ROOT,
                #             "Output",
                #             "b_Soft_WL_Kernel_random_split",
                #             "Jackson",
                #             "Matched_SoftWL_dict_iter_"
                #             + str(iteration)
                #             + "_PhenoGraph_k_"
                #             + str(PhenoGraph_k)
                #             + "_"
                #             + node_label
                #             + "_centroid_alignment.pkl",
                #         ),
                #         "rb",
                #     ),
                # )
                # Patient_IDs_jackson = SoftWL_dict_jackson["Patient_id"]
                # Histograms_jackson = SoftWL_dict_jackson["Histogram"]
                # Histograms_jackson = Histograms_jackson / np.sum(
                #     Histograms_jackson, axis=1, keepdims=True
                # )
                # clinical = pd.read_csv(
                #     os.path.join(
                #         PROJECT_ROOT,
                #         "Input",
                #         "Clinical",
                #         "Jackson",
                #         "clinical.csv",
                #     )
                # )
                # clinical['Overall Survival Status'] = clinical['Overall Survival Status'].map({'0:LIVING':0, '1:DECEASED':1})
                # clinical['Relapse Free Status'] = clinical['Relapse Free Status'].map({'0:Not Recurred':0, '1:Recurred':1})


                # Length_jackson = [
                #     clinical.loc[
                #         clinical["patient_id"] == i, "Overall Survival (Months)"
                #     ].values[0]
                #     for i in Patient_IDs_jackson
                # ]
                # Status_jackson = [
                #     clinical.loc[
                #         clinical["patient_id"] == i, "Overall Survival Status"
                #     ].values[0]
                #     for i in Patient_IDs_jackson
                # ]

                # DF_jackson = pd.DataFrame(
                #     {"Length": Length_jackson, "Status": Status_jackson}
                # ).dropna()
                # Length_jackson_ = np.array(DF_jackson["Length"])
                # Status_jackson_ = np.array(DF_jackson["Status"])
                # Histograms_jackson_ = Histograms_jackson[DF_jackson.index.values, :]
                # Patient_IDs_jackson_ = np.array(Patient_IDs_jackson)[
                #     DF_jackson.index.values
                # ]
                # DF = pd.DataFrame(
                #     {
                #         "length": Length_jackson_,
                #         "status": Status_jackson_,
                #     }
                # )
                # for subgroup_id, Candidates in Overpresented_patterns.items():
                #     for pattern_id in Candidates:
                #         histogram = Histograms_jackson_[:, pattern_id]
                #         DF["pattern_" + str(pattern_id + 1)] = histogram > 0
                #         if np.sum(histogram > 0) == 0:
                #             continue
                #         cph = CoxPHFitter()
                #         cph.fit(
                #                 DF,
                #                 duration_col="length",
                #                 event_col="status",
                #                 formula="pattern_" + str(pattern_id + 1),
                #             )
                #         hr = cph.hazard_ratios_["pattern_" + str(pattern_id + 1)]
                #         p = cph.summary["p"]["pattern_" + str(pattern_id + 1)]
                #         if p < 0.05 / Num_patterns_to_test:
                #                 print(
                #                     "[Jackson, Overall, Existence] Pattern: {}, subgroup_id: {}, HR: {}, p: {}".format(
                #                         pattern_id + 1, subgroup_id, hr, p
                #                     )
                #                 )
                 


