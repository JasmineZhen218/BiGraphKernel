import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import PROJECT_ROOT
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import networkx as nx
from matplotlib.lines import Line2D
import pandas as pd
from c_Population_graph.population_graph import construct_PopulationGraph
from c_Population_graph.community_detection import detect_communities
from e_Survival_analysis.survival_analysis import calculate_hazard_ratio
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from overpresented_patterns import find_overpresented_patterns

iteration = 2
PhenoGraph_k =  30
size_smallest_cluster = 20
survival_type = 'Overall'
PopulationGraph_type = 'knn_graph'
para_dict = {
    'weight_threshold_percentile': 95,
    'knn_k':15
}

SoftWL_dict= pickle.load(
    open(
        os.path.join(
            PROJECT_ROOT,
            "Output",
            "b_Soft_WL_Kernel",
            "Danenberg",
            "Cohort_1",
            "SoftWL_dict_iter_"
            + str(iteration)
            + "_PhenoGraph_k_"
            + str(PhenoGraph_k)
            + ".pkl",
        ),
        "rb",
    ),
)
Patient_IDs = SoftWL_dict["Patient_id"]
Gram_matrix = SoftWL_dict["Gram_matrix"]
Histograms = SoftWL_dict["Histogram"]
clinical = pd.read_csv(
    os.path.join(
        PROJECT_ROOT,
        "Input",
        "Clinical",
        "Danenberg",
        "clinical.csv",
    )
)
clinical['Overall Survival Status'] = clinical['Overall Survival Status'].map({'0:LIVING':0, '1:DECEASED':1})
clinical['Relapse Free Status'] = clinical['Relapse Free Status'].map({'0:Not Recurred':0, '1:Recurred':1})
clinical['Disease-specific Survival Status'] = clinical['Disease-specific Survival Status'].map({'0:LIVING':0, '1:DECEASED':1})

G_population = construct_PopulationGraph(
    Gram_matrix,
    PopulationGraph_type,
    para_dict = para_dict,
)
Community_ids = detect_communities(G_population, size_smallest_cluster)

if survival_type == 'Overall':
    Length = [clinical.loc[clinical['patient_id'] == i, 'Overall Survival (Months)'].values[0] for i in Patient_IDs]
    Status = [clinical.loc[clinical['patient_id'] == i, 'Overall Survival Status'].values[0] for i in Patient_IDs]  
elif survival_type == 'Relpase-free':
    Length = [clinical.loc[clinical['patient_id'] == i, 'Relapse Free Status (Months)'].values[0] for i in Patient_IDs]
    Status = [clinical.loc[clinical['patient_id'] == i, 'Relapse Free Status'].values[0] for i in Patient_IDs] 
elif survival_type == 'Disease-specific':
    Length = [clinical.loc[clinical['patient_id'] == i, 'Disease-specific Survival (Months)'].values[0] for i in Patient_IDs]
    Status = [clinical.loc[clinical['patient_id'] == i, 'Disease-specific Survival Status'].values[0] for i in Patient_IDs]
DF = pd.DataFrame({"Length": Length, "Status": Status, "Community_ids": Community_ids}).dropna()
Length_ = np.array(DF["Length"])
Status_ = np.array(DF["Status"])
Community_ids_ = np.array(DF["Community_ids"])
HR = calculate_hazard_ratio(Length_, Status_, Community_ids_)
HR = sorted(HR, key=lambda x: x["hr"], reverse=True)
Subgroup_ids = np.zeros_like(Community_ids)
Subgroup_ids_ = np.zeros_like(Community_ids_)
for i in range(len(HR)):
    Subgroup_ids[Community_ids == HR[i]["community_id"]] = i + 1
    Subgroup_ids_[Community_ids_ == HR[i]["community_id"]] = i + 1
    HR[i]["subgroup_id"] = i + 1
for i in range(len(HR)):
    print("S{}:".format(HR[i]['subgroup_id']))
    print("Survival: hr = {}, p = {}".format(HR[i]["hr"], HR[i]["p"]))

Overpresented_patterns = find_overpresented_patterns(Histograms, Subgroup_ids, 3)
Prognosis_relevant_patterns_candidates = []
Prognosis_relevant_patterns_candidates_subgroup_ids = []
for hr_dict in HR:
    subgroup_id = hr_dict['subgroup_id']
    hr = hr_dict["hr"]
    p = hr_dict["p"]
    if p<0.05:
        over_presented_patterns = Overpresented_patterns[subgroup_id]
        Prognosis_relevant_patterns_candidates.extend(over_presented_patterns)
        Prognosis_relevant_patterns_candidates_subgroup_ids.extend([subgroup_id]*len(over_presented_patterns))
print(Prognosis_relevant_patterns_candidates)
print(Prognosis_relevant_patterns_candidates_subgroup_ids)


Histograms_candidates = Histograms[:, np.array(Prognosis_relevant_patterns_candidates)]
Histograms_candidates = (Histograms_candidates - np.mean(Histograms_candidates, axis = 0))/np.std(Histograms_candidates, axis = 0)
DF = pd.DataFrame({})
DF['patient_id'] = Patient_IDs
DF['length'] = Length
DF['status'] = Status
for i in range(len(Prognosis_relevant_patterns_candidates)):
    DF['pattern_{}'.format(Prognosis_relevant_patterns_candidates[i])] = Histograms_candidates[:, i]
DF = DF.dropna()
print(DF.shape)
DF.head()

from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(DF, duration_col='length', event_col='status',
        formula = " + ".join(['pattern_'+str(i) for i in Prognosis_relevant_patterns_candidates])) # use a formula helps encode categorical variables
for i in range(len(Prognosis_relevant_patterns_candidates)):
    pattern_id = Prognosis_relevant_patterns_candidates[i]
    subgroup_id = Prognosis_relevant_patterns_candidates_subgroup_ids[i]
    p = cph.summary["p"]['pattern_'+str(pattern_id)]
    hr = cph.hazard_ratios_['pattern_'+str(pattern_id)]
    if p < 0.05:
        print("Pattern {} overpresented in S{} is associated with survival (p = {:.5f}, HR = {:.5f})".format(pattern_id,subgroup_id, p, hr))

pattern_id = 0
histogram_pattern = DF["pattern_" + str(pattern_id)].values
kmf = KaplanMeierFitter()
f, ax = plt.subplots(figsize=(5, 5))
length_A, event_observed_A = (
    Length_[histogram_pattern > np.percentile(histogram_pattern, 75)],
    Status_[histogram_pattern > np.percentile(histogram_pattern, 75)],
)
label = "Highest 25% (N={})".format(
    np.sum(histogram_pattern > np.percentile(histogram_pattern, 75))
)
kmf.fit(length_A, event_observed_A, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette[i],
    show_censors=True,
    censor_styles={"ms": 3, "marker": "s"},
)
print(histogram_pattern)
print(histogram_pattern < np.percentile(histogram_pattern, 25))
length_B, event_observed_B = (
    Length_[histogram_pattern < np.percentile(histogram_pattern, 25)],
    Status_[histogram_pattern < np.percentile(histogram_pattern, 25)],
)
label = "Lowerest 25% (N={})".format(
    np.sum(histogram_pattern < np.percentile(histogram_pattern, 25))
)
kmf.fit(length_B, event_observed_B, label=label)
kmf.plot_survival_function(
    ax=ax,
    ci_show=False,
    color=color_palette[i],
    show_censors=True,
    censor_styles={"ms": 3, "marker": "s"},
)
# log_rank_test  = multivariate_logrank_test(
#         Length_[Subgroup_ids_!=0], Subgroup_ids_[Subgroup_ids_!=0],Status_[Subgroup_ids_!=0]
#     )
# p_value = log_rank_test.p_value
# ax.legend(ncol=2, fontsize=10)
# ax.set_title("p-value = {:.5f}".format(p_value), fontsize=12)
ax.set_xlabel("Time (Month)", fontsize=12)
ax.set_ylabel("Survival (%)", fontsize=12)
ax.set(
    ylim=(-0.05, 1.05),
)
sns.despine()