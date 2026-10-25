"""
BraTS Multi-Parametric Dataset & DataLoader Pipeline.
Loads multi-channel 3D MRI scans (T1, T1ce, T2, FLAIR), applies standardization,
extracts nested labels (ET, TC, WT), and provides training/evaluation patch generators.
"""

import os
import json
import glob
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.data.dicom_io import DicomSeriesReader
from src.data.transforms import (
    nonzero_zscore_normalize,
    extract_brain_mask,
    resample_volume_3d,
    rigid_co_register_3d,
    brats_labels_to_nested_regions,
    crop_or_pad_3d
)


class BraTSCase:
    """Represents a single clinical patient case with 4 MRI sequences and optional annotations."""
    def __init__(
        self,
        case_id: str,
        t1_path: str,
        t1ce_path: str,
        t2_path: str,
        flair_path: str,
        seg_path: Optional[str] = None,
        class_label: int = 0
    ):
        self.case_id = case_id
        self.paths = {
            "t1": t1_path,
            "t1ce": t1ce_path,
            "t2": t2_path,
            "flair": flair_path,
            "seg": seg_path
        }
        self.class_label = class_label


class BraTSDataset:
    """
    Volumetric dataset supporting:
    - 4-channel tensor assembly (T1, T1ce, T2, FLAIR)
    - Multimodal co-registration & brain-only non-zero Z-score normalization
    - 3-channel nested ground-truth segmentation masks (ET, TC, WT)
    - Classification targets: Glioma II/III (0), Glioblastoma IV (1), Meningioma (2)
    - Random/Deterministic 3D patch extraction (128x128x128)
    """

    def __init__(
        self,
        cases: List[BraTSCase],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        is_training: bool = True,
        co_register: bool = False # set False if pre-registered (e.g. standard BraTS)
    ):
        self.cases = cases
        self.patch_size = patch_size
        self.is_training = is_training
        self.co_register = co_register

    def __len__(self) -> int:
        return len(self.cases)

    def load_case_tensor(self, case: BraTSCase) -> Tuple[np.ndarray, Optional[np.ndarray], int, Dict[str, Any]]:
        """
        Loads and standardizes all 4 modalities for a given patient case.
        Returns:
            image: (4, H, W, D) float32 normalized tensor
            label: (3, H, W, D) float32 binary nested mask or None
            class_label: int
            meta: dictionary with spatial metadata
        """
        # Load T1ce as the fixed reference frame
        t1ce, meta = DicomSeriesReader.load_nifti_volume(case.paths["t1ce"])
        t1, _ = DicomSeriesReader.load_nifti_volume(case.paths["t1"])
        t2, _ = DicomSeriesReader.load_nifti_volume(case.paths["t2"])
        flair, _ = DicomSeriesReader.load_nifti_volume(case.paths["flair"])

        if self.co_register:
            t1, _ = rigid_co_register_3d(t1, t1ce)
            t2, _ = rigid_co_register_3d(t2, t1ce)
            flair, _ = rigid_co_register_3d(flair, t1ce)

        # Apply non-zero Z-score normalization on each channel independently
        t1 = nonzero_zscore_normalize(t1)
        t1ce = nonzero_zscore_normalize(t1ce)
        t2 = nonzero_zscore_normalize(t2)
        flair = nonzero_zscore_normalize(flair)

        # Stack into 4D tensor: Channel order [T1, T1ce, T2, FLAIR]
        image_4d = np.stack([t1, t1ce, t2, flair], axis=0).astype(np.float32)

        # Load segmentation if available
        nested_label = None
        if case.paths["seg"] and os.path.exists(case.paths["seg"]):
            raw_seg, _ = DicomSeriesReader.load_nifti_volume(case.paths["seg"])
            nested_label = brats_labels_to_nested_regions(raw_seg)

        return image_4d, nested_label, case.class_label, meta

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self.cases[idx]
        image_4d, label_3d, class_label, meta = self.load_case_tensor(case)

        if self.is_training:
            # Crop random patch centered around brain or tumor
            C, H, W, D = image_4d.shape
            ph, pw, pd = self.patch_size

            # Ensure valid bounds
            max_h = max(0, H - ph)
            max_w = max(0, W - pw)
            max_d = max(0, D - pd)

            # Center sampling around foreground/tumor with 75% probability if label exists
            if label_3d is not None and np.random.rand() < 0.75 and np.any(label_3d[2] > 0):
                # Sample center from whole tumor
                tumor_indices = np.argwhere(label_3d[2] > 0)
                center = tumor_indices[np.random.choice(len(tumor_indices))]
                sh = np.clip(center[0] - ph // 2, 0, max_h)
                sw = np.clip(center[1] - pw // 2, 0, max_w)
                sd = np.clip(center[2] - pd // 2, 0, max_d)
            else:
                sh = np.random.randint(0, max_h + 1) if max_h > 0 else 0
                sw = np.random.randint(0, max_w + 1) if max_w > 0 else 0
                sd = np.random.randint(0, max_d + 1) if max_d > 0 else 0

            image_patch = crop_or_pad_3d(image_4d[:, sh:sh+ph, sw:sw+pw, sd:sd+pd], self.patch_size)
            label_patch = (
                crop_or_pad_3d(label_3d[:, sh:sh+ph, sw:sw+pw, sd:sd+pd], self.patch_size)
                if label_3d is not None else None
            )

            # Random 3D flips for data augmentation
            if np.random.rand() < 0.5:
                image_patch = np.flip(image_patch, axis=1).copy()
                if label_patch is not None:
                    label_patch = np.flip(label_patch, axis=1).copy()
            if np.random.rand() < 0.5:
                image_patch = np.flip(image_patch, axis=2).copy()
                if label_patch is not None:
                    label_patch = np.flip(label_patch, axis=2).copy()

            return {
                "case_id": case.case_id,
                "image": image_patch,
                "label": label_patch,
                "class_label": class_label,
                "meta": meta
            }
        else:
            return {
                "case_id": case.case_id,
                "image": image_4d,
                "label": label_3d,
                "class_label": class_label,
                "meta": meta
            }
