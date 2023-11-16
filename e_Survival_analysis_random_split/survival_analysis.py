import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def calculate_hazard_ratio(length, status, community_id, adjust_dict = {}):
    """
    Calculate hazard ratio for each community
    :param length: length of follow-up
    :param status: status of follow-up
    :param community_id: community id
    :return: HR: hazard ratio for each community
    """
    # Calculate hazard ratio
    HR = []
    cph = CoxPHFitter()
    unique_community_id = np.unique(community_id[community_id != 0])
    for i in unique_community_id:
        DF = pd.DataFrame(
                {"length": length, "status": status, "community": community_id == i}
            )
        for key, value in adjust_dict.items():
            DF[key] = value
        cph.fit(
            DF,
            duration_col="length",
            event_col="status",
            show_progress=False,
        )
        HR.append(
            {
                "community_id": i,
                # "status": np.array(status)[community_id == i],
                # "length": np.array(length)[community_id == i],
                "hr": cph.hazard_ratios_["community"],
                "hr_lower": np.exp(
                    cph.confidence_intervals_["95% lower-bound"]["community"]
                ),
                "hr_upper": np.exp(
                    cph.confidence_intervals_["95% upper-bound"]["community"]
                ),
                "p": cph.summary["p"]["community"],
            }
        )
    return HR