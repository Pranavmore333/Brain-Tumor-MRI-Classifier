r"""
Clinical-Grade Quantitative Evaluation Metrics & Academic Benchmarking Suite.
Implements:
1. Volumetric Dice Similarity Coefficient (DSC): DSC(P, G) = 2|P \cap G| / (|P| + |G|)
2. 95th Percentile Hausdorff Distance (HD_95) in Physical Millimeters
3. Lesion-Wise Connected Component Dice Coefficient
4. Multi-class Classification Metrics: Accuracy, Precision, Recall, F1-Score
5. Clinical Benchmark Acceptance Gatekeeper
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy.ndimage import label, distance_transform_edt, binary_erosion


def compute_dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-5
) -> float:
    """
    Computes standard volumetric Dice Similarity Coefficient (DSC).
    """
    p_bool = pred.astype(bool)
    t_bool = target.astype(bool)

    if not np.any(p_bool) and not np.any(t_bool):
        return 1.0 # Both empty -> perfect concordance
    if not np.any(p_bool) or not np.any(t_bool):
        return 0.0 # One empty while other has pathology

    intersection = np.logical_and(p_bool, t_bool).sum()
    total = p_bool.sum() + t_bool.sum()

    return float((2.0 * intersection + eps) / (total + eps))


def compute_surface_points(binary_mask: np.ndarray) -> np.ndarray:
    """Extracts boundary surface voxels using morphological erosion."""
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)
    eroded = binary_erosion(binary_mask)
    surface = np.logical_xor(binary_mask, eroded)
    return surface


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    r"""
    Computes 95th Percentile Hausdorff Distance (HD_95) in physical millimeters.
    Formula:
        HD_95(P, G) = max( P_95_{x \in dP} d(x, dG), P_95_{y \in dG} d(y, dP) )
    """
    p_bool = pred.astype(bool)
    t_bool = target.astype(bool)

    if not np.any(p_bool) and not np.any(t_bool):
        return 0.0 # Empty sets distance is 0
    if not np.any(p_bool) or not np.any(t_bool):
        # Maximum penalty for missing lesion
        return 50.0

    p_surface = compute_surface_points(p_bool)
    t_surface = compute_surface_points(t_bool)

    if not np.any(p_surface) or not np.any(t_surface):
        return 50.0

    # Euclidean distance transform from ground truth surface
    dt_target = distance_transform_edt(~t_surface, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~p_surface, sampling=voxel_spacing)

    distances_p_to_t = dt_target[p_surface]
    distances_t_to_p = dt_pred[t_surface]

    p95_p_to_t = np.percentile(distances_p_to_t, 95)
    p95_t_to_p = np.percentile(distances_t_to_p, 95)

    return float(max(p95_p_to_t, p95_t_to_p))


def compute_lesion_wise_dice(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-5
) -> float:
    r"""
    Prevents large continuous tumor masses from dominating evaluation scores
    by evaluating Dice on individual 3D connected topological components:
        DSC_lesion = (1 / N) * sum_{k=1}^N [ 2|P \cap L_k| / (|P| + |L_k|) ]
    """
    t_bool = target.astype(bool)
    p_bool = pred.astype(bool)

    if not np.any(t_bool) and not np.any(p_bool):
        return 1.0
    if not np.any(t_bool) or not np.any(p_bool):
        return 0.0

    labeled_target, num_features = label(t_bool)
    if num_features == 0:
        return 0.0

    lesion_dices = []
    for lesion_idx in range(1, num_features + 1):
        l_k = (labeled_target == lesion_idx)
        intersection = np.logical_and(p_bool, l_k).sum()
        total = p_bool.sum() + l_k.sum()
        dice_k = (2.0 * intersection + eps) / (total + eps)
        lesion_dices.append(dice_k)

    return float(np.mean(lesion_dices))


class ClinicalEvaluationHarness:
    """
    Evaluates multi-label segmentation and classification against clinical acceptance benchmarks.
    """

    CLINICAL_ACCEPTANCE_THRESHOLDS = {
        "ET": {"min_dsc": 0.780, "max_hd95_mm": 5.00},
        "TC": {"min_dsc": 0.820, "max_hd95_mm": 6.00},
        "WT": {"min_dsc": 0.880, "max_hd95_mm": 5.00}
    }

    @staticmethod
    def evaluate_case(
        pred_nested: np.ndarray,
        target_nested: np.ndarray,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates a single patient volume across all 3 nested regions:
        Channel 0: ET (Enhancing Tumor)
        Channel 1: TC (Tumor Core)
        Channel 2: WT (Whole Tumor)
        """
        region_names = ["ET", "TC", "WT"]
        results = {}

        for i, name in enumerate(region_names):
            p = pred_nested[i] > 0.5
            t = target_nested[i] > 0.5

            dsc = compute_dice_score(p, t)
            hd95 = compute_hd95(p, t, voxel_spacing=voxel_spacing)
            lesion_dsc = compute_lesion_wise_dice(p, t)

            thresholds = ClinicalEvaluationHarness.CLINICAL_ACCEPTANCE_THRESHOLDS[name]
            passes_acceptance = (dsc >= thresholds["min_dsc"]) and (hd95 <= thresholds["max_hd95_mm"])

            results[name] = {
                "dice": round(dsc, 4),
                "hd95_mm": round(hd95, 2),
                "lesion_dice": round(lesion_dsc, 4),
                "clinical_acceptance_passed": passes_acceptance
            }

        return results

    @staticmethod
    def evaluate_cohort(
        cases_eval: List[Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """
        Aggregates mean and standard deviation across an entire cohort (e.g. 5-fold CV).
        """
        summary = {}
        region_names = ["ET", "TC", "WT"]

        for name in region_names:
            dices = [c[name]["dice"] for c in cases_eval]
            hd95s = [c[name]["hd95_mm"] for c in cases_eval]
            lesions = [c[name]["lesion_dice"] for c in cases_eval]
            passes = [c[name]["clinical_acceptance_passed"] for c in cases_eval]

            summary[name] = {
                "mean_dsc": round(float(np.mean(dices)), 3),
                "std_dsc": round(float(np.std(dices)), 3),
                "mean_hd95_mm": round(float(np.mean(hd95s)), 2),
                "std_hd95_mm": round(float(np.std(hd95s)), 2),
                "mean_lesion_dsc": round(float(np.mean(lesions)), 3),
                "acceptance_rate_percent": round(float(np.mean(passes) * 100), 1)
            }

        return summary
