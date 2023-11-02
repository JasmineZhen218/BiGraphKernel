import numpy as np


def calculate_relative_presentation(Histogram, Subgroup_ids, threshold):
    """
    Find patterns that are overpresented in the subgroup compared to the whole population.
    Parameters
    ----------
    Histogram : ndarray
        Histogram of the patterns of each patient. shape = (n_patients, n_patterns)
    Subgroup_ids : ndarray
        Ids of the patients in the subgroup. shape = (n_patients)
    threshold : float
        Threshold to consider a pattern overpresented in the subgroup.
    Returns
    -------
    overpresented_patterns : dict
        List of the patterns that are overpresented in the subgroup.
    """
    # Find the patterns that are overpresented in the subgroup compared to the whole population.
    median_presentation = np.median(Histogram, axis=0)
    Relative_presentation = {}
    unique_subgroup_ids = np.unique(Subgroup_ids[Subgroup_ids != 0])
    for subgroup_id in unique_subgroup_ids:
        Histogram_intra_group = Histogram[Subgroup_ids == subgroup_id]
        median_presentation_intra_group = np.median(Histogram_intra_group, axis=0)
        relative_presentation = median_presentation_intra_group / median_presentation
        Relative_presentation[subgroup_id] = relative_presentation
    return Relative_presentation
