

def get_node_id(dataset, node_label):
    if dataset == "Danenberg":
        if node_label == "CellType":
            cell_phenotype_id = {
                "CK^{med}ER^{lo}": 0,
                "ER^{hi}CXCL12^{+}": 1,
                "CK^{lo}ER^{lo}": 2,
                "Ep Ki67^{+}": 3,
                "CK^{lo}ER^{med}": 4,
                "Ep CD57^{+}": 5,
                "MHC I & II^{hi}": 6,
                "Basal": 7,
                "HER2^{+}": 8,
                "CK8-18^{hi}CXCL12^{hi}": 9,
                "CK^{+} CXCL12^{+}": 10,
                "CK8-18^{hi}ER^{lo}": 11,
                "CK8-18^{+} ER^{hi}": 12,
                "MHC I^{hi}CD57^{+}": 13,
                "MHC^{hi}CD15^{+}": 14,
                "CD15^{+}": 15,
                "CD4^{+} T cells & APCs": 16,
                "CD4^{+} T cells": 17,
                "CD8^{+} T cells": 18,
                "T_{Reg} & T_{Ex}": 19,
                "B cells": 20,
                "CD38^{+} lymphocytes": 21,
                "Macrophages": 22,
                "Granulocytes": 23,
                "Macrophages & granulocytes": 24,
                "CD57^{+}": 25,
                "Ki67^{+}": 26,
                "Endothelial": 27,
                "Fibroblasts": 28,
                "Fibroblasts FSP1^{+}": 29,
                "Myofibroblasts PDPN^{+}": 30,
                "Myofibroblasts": 31,
            }
        elif node_label == "CellCategory":
            cell_phenotype_id = {
                "tumor": 0,
                "immune": 1,
                "stromal": 2,
            }
        elif node_label == "TMECellType":
            cell_phenotype_id = {
                "tumor": 0,
                "CD4^{+} T cells & APCs": 1,
                "CD4^{+} T cells": 2,
                "CD8^{+} T cells": 3,
                "T_{Reg} & T_{Ex}": 4,
                "B cells": 5,
                "CD38^{+} lymphocytes": 6,
                "Macrophages": 7,
                "Granulocytes": 8,
                "Macrophages & granulocytes": 9,
                "CD57^{+}": 10,
                "Ki67^{+}": 11,
                "Endothelial": 12,
                "Fibroblasts": 13,
                "Fibroblasts FSP1^{+}": 14,
                "Myofibroblasts PDPN^{+}": 15,
                "Myofibroblasts": 16,
            }
    elif dataset == "Jackson":
        cell_phenotype_id = {
            "B cells": 0,
            "B and T cells": 1,
            "T cells_1": 2,
            "Macrophages_1": 3,
            "T cells_2": 4,
            "Macrophages_2": 5,
            "Endothelial": 6,
            "Vimentin hi": 7,
            "small circular": 8,
            "small elongated": 9,
            "Fibronectin hi": 10,
            "larger elongated": 11,
            "SMA hi Vimentin": 12,
            "hypoxic": 13,
            "apoptotic": 14,
            "proliferative": 15,
            "p53 EGFR": 16,
            "Basal CK": 17,
            "CK7 CK hi Cadherin": 18,
            "CK7 CK": 19,
            "Epithelial low": 20,
            "CK low HR low": 21,
            "HR hi CK": 22,
            "CK HR": 23,
            "HR low CK": 24,
            "CK low HR hi p53": 25,
            "Myoepithalial": 26,
        }
    return cell_phenotype_id

def get_node_color(dataset, node_label):
    if dataset == "Danenberg":
        if node_label == "CellType":
            cell_color = {
                "CK^{med}ER^{lo}": "#40647A",
                "ER^{hi}CXCL12^{+}": "#99CCCC",
                "CD4^{+} T cells & APCs": "#F8B195",
                "CD4^{+} T cells": "#FF50A2",
                "Endothelial": "#FFC400",
                "Fibroblasts": "#007B1D",
                "Myofibroblasts PDPN^{+}": "#c6ce2b",
                "CD8^{+} T cells": "#D6316F",
                "CK8-18^{hi}CXCL12^{hi}": "#00BFFF",
                "Myofibroblasts": "#81ca33",
                "CK^{lo}ER^{lo}": "#73A9DF",
                "Macrophages": "#800080",
                "CK^{+} CXCL12^{+}": "#1E90FF",
                "CK8-18^{hi}ER^{lo}": "#006BD7",
                "CK8-18^{+} ER^{hi}": "#005A9C",
                "CD15^{+}": "#D4CFC9",
                "MHC I & II^{hi}": "#8F756B",
                "T_{Reg} & T_{Ex}": "#FE7A15",
                "CD57^{+}": "#D699B4",
                "Ep Ki67^{+}": "#000066",
                "CK^{lo}ER^{med}": "#0088B2",
                "Macrophages & granulocytes": "#d6bbf6",
                "CD38^{+} lymphocytes": "#FBFEC9",
                "Ki67^{+}": "#006400",
                "HER2^{+}": "#D3E7EE",
                "B cells": "#FFFF00",
                "Basal": "#805B3B",
                "Fibroblasts FSP1^{+}": "#37AD3F",
                "Granulocytes": "#6C5B7B",
                "MHC I^{hi}CD57^{+}": "#A69287",
                "Ep CD57^{+}": "#708090",
                "MHC^{hi}CD15^{+}": "#F4A460",
            }
        elif node_label == "CellCategory":
            cell_color = {
                "tumor": "blue",
                "immune": "red",
                "stroma": "green",
            }
        elif node_label == "TMECellType":
            cell_color = {
                "tumor": "blue",
                "CD4^{+} T cells & APCs": "#F8B195",
                "CD4^{+} T cells": "#FF50A2",
                "Endothelial": "#FFC400",
                "Fibroblasts": "#007B1D",
                "Myofibroblasts PDPN^{+}": "#c6ce2b",
                "CD8^{+} T cells": "#D6316F",
                "Myofibroblasts": "#81ca33",
                "Macrophages": "#800080",
                "CD15^{+}": "#D4CFC9",
                "T_{Reg} & T_{Ex}": "#FE7A15",
                "CD57^{+}": "#D699B4",
                "Macrophages & granulocytes": "#d6bbf6",
                "CD38^{+} lymphocytes": "#FBFEC9",
                "Ki67^{+}": "#006400",
                "B cells": "#FFFF00",
                "Fibroblasts FSP1^{+}": "#37AD3F",
                "Granulocytes": "#6C5B7B",
            }
    return cell_color


def get_paired_markers(source="Danenberg", target="Jackson"):
    if (source == "Danenberg") and (target == "Jackson"):
        Paired_Markers = [
            ("panCK", "AE1/AE3"),
            ("CK8-18", "CK8/18"),
            ("CK5", "CK5"),
            ("ER", "ER"),
            ("HER2 (3B5)", "Her2"),
            ("HER2 (D8F12)", "Her2"),
            ("CD31-vWF", "vWF/CD31"),
            ("SMA", "SMA"),
            ("Ki-67", "Ki-67"),
            ("c-Caspase3", "cleaved PARP/cleaved Caspase3"),
            ("CD45RA", "CD45"),
            ("CD3", "CD3"),
            ("CD20", "CD20"),
            ("CD68", "CD68"),
            ("Histone H3", "Histone H3_1"),
            ("Histone H3", "Histone H3_2"),
            ("DNA1", "DNA1"),
        ]
    return Paired_Markers
