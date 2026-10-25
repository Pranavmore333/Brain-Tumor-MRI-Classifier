"""
DICOM & NIfTI I/O and Series Parsing Module.
Handles:
- Native 2D DICOM slice series parsing and 3D volume reconstruction using SimpleITK
- Extracting affine geometry (origin, spacing, direction cosines)
- Multi-parametric sequence verification (T1, T1ce, T2, FLAIR)
- Conversion between ITK Image objects and NumPy tensors
- Generating synthetic clinical multi-parametric MRI datasets for unit tests and benchmarks
"""

import os
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import SimpleITK as sitk


class DicomSeriesReader:
    """
    Parses and reconstructs 2D DICOM slice series into continuous 3D volumes.
    Preserves anatomical coordinates, spacing, and spatial orientation.
    """

    @staticmethod
    def load_dicom_series(series_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reads a directory containing 2D DICOM slice files and reconstructs a 3D volume.
        Returns:
            volume_np: (H, W, D) float32 numpy array
            metadata: dictionary containing spacing, origin, direction cosines, and tags
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
        if not dicom_names:
            raise FileNotFoundError(f"No DICOM slice series found in {series_dir}")
            
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        itk_image = reader.Execute()
        
        spacing = itk_image.GetSpacing()
        origin = itk_image.GetOrigin()
        direction = itk_image.GetDirection()
        size = itk_image.GetSize()
        
        # sitk.GetArrayFromImage returns (Z, Y, X) -> transpose to (X, Y, Z) / (H, W, D)
        volume_np = sitk.GetArrayFromImage(itk_image)
        volume_np = np.transpose(volume_np, (2, 1, 0)).astype(np.float32)
        
        metadata = {
            "spacing": spacing,
            "origin": origin,
            "direction": direction,
            "size": size,
            "num_slices": len(dicom_names)
        }
        return volume_np, metadata

    @staticmethod
    def load_nifti_volume(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads a .nii or .nii.gz NIfTI volume using SimpleITK.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NIfTI file not found: {filepath}")
            
        itk_image = sitk.ReadImage(filepath)
        spacing = itk_image.GetSpacing()
        origin = itk_image.GetOrigin()
        direction = itk_image.GetDirection()
        
        volume_np = sitk.GetArrayFromImage(itk_image)
        volume_np = np.transpose(volume_np, (2, 1, 0)).astype(np.float32)
        
        metadata = {
            "spacing": spacing,
            "origin": origin,
            "direction": direction,
            "shape": volume_np.shape
        }
        return volume_np, metadata

    @staticmethod
    def save_nifti_volume(
        volume_np: np.ndarray,
        filepath: str,
        reference_metadata: Optional[Dict[str, Any]] = None,
        is_label: bool = False
    ) -> str:
        """
        Saves a NumPy volume as a .nii.gz file with preserved physical geometry.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Transpose (H, W, D) -> (D, W, H) for SimpleITK GetImageFromArray
        transposed = np.transpose(volume_np, (2, 1, 0))
        if is_label:
            transposed = transposed.astype(np.uint8)
            itk_image = sitk.GetImageFromArray(transposed, isVector=False)
        else:
            transposed = transposed.astype(np.float32)
            itk_image = sitk.GetImageFromArray(transposed, isVector=False)

        if reference_metadata:
            itk_image.SetSpacing(reference_metadata.get("spacing", (1.0, 1.0, 1.0)))
            itk_image.SetOrigin(reference_metadata.get("origin", (0.0, 0.0, 0.0)))
            itk_image.SetDirection(reference_metadata.get("direction", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)))
        else:
            itk_image.SetSpacing((1.0, 1.0, 1.0))
            itk_image.SetOrigin((0.0, 0.0, 0.0))

        sitk.WriteImage(itk_image, filepath)
        return filepath


def generate_synthetic_brats_case(
    output_dir: str,
    case_id: str = "BraTS_SYN_001",
    shape: Tuple[int, int, int] = (128, 128, 128),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tumor_type: str = "Glioblastoma Grade IV"
) -> Dict[str, str]:
    """
    Generates an anatomically structured synthetic multi-parametric MRI case
    with all 4 modalities (T1, T1ce, T2, FLAIR) and ground-truth nested tumor sub-regions.
    Useful for testing pipelines without requiring external 100GB BraTS downloads.
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W, D = shape
    
    # 1. Create ellipsoidal Brain Parenchyma background
    x, y, z = np.indices((H, W, D))
    cx, cy, cz = H // 2, W // 2, D // 2
    rx, ry, rz = H * 0.40, W * 0.35, D * 0.40
    brain_mask = ((x - cx)**2 / rx**2 + (y - cy)**2 / ry**2 + (z - cz)**2 / rz**2) <= 1.0

    # 2. Add realistic tumor pathology
    # Place tumor in right frontal/temporal lobe
    tcx, tcy, tcz = int(cx + H * 0.12), int(cy + W * 0.10), int(cz + D * 0.05)
    
    # Whole Tumor / Edema (Radius ~ 18 voxels)
    dist_sq = (x - tcx)**2 + (y - tcy)**2 + (z - tcz)**2
    edema_mask = (dist_sq <= 18**2) & brain_mask
    
    # Tumor Core (Radius ~ 12 voxels)
    tc_mask = (dist_sq <= 12**2) & brain_mask
    
    # Active Enhancing Tumor (Outer ring between 7 and 12 voxels)
    et_mask = (dist_sq <= 12**2) & (dist_sq > 7**2) & brain_mask
    
    # Necrotic Core (NCR inside radius 7)
    ncr_mask = (dist_sq <= 7**2) & brain_mask
    
    # Create BraTS discrete ground truth:
    # 0: background, 1: NCR/NETC, 2: ED, 4: ET
    seg = np.zeros(shape, dtype=np.uint8)
    seg[edema_mask] = 2
    seg[ncr_mask] = 1
    seg[et_mask] = 4

    # 3. Simulate mpMRI Modality contrasts:
    # Base brain intensity
    base_brain = np.zeros(shape, dtype=np.float32)
    base_brain[brain_mask] = 100.0 + 10.0 * np.sin(x[brain_mask] * 0.1)

    # T1: Brain is grey, edema is hypointense, necrotic is dark
    t1 = base_brain.copy()
    t1[edema_mask] -= 30.0
    t1[ncr_mask] -= 50.0
    t1 += np.random.normal(0, 5, shape)
    t1[~brain_mask] = 0

    # T1ce: Contrast enhancement makes ET brightly enhancing, NCR dark
    t1ce = base_brain.copy()
    t1ce[edema_mask] -= 20.0
    t1ce[ncr_mask] -= 45.0
    t1ce[et_mask] += 90.0  # Hyper-intense enhancement
    t1ce += np.random.normal(0, 5, shape)
    t1ce[~brain_mask] = 0

    # T2: Edema and necrotic core are hyperintense (fluid sensitive)
    t2 = base_brain.copy()
    t2[edema_mask] += 70.0  # Bright edema
    t2[ncr_mask] += 80.0    # Bright necrotic fluid
    t2[et_mask] += 30.0
    t2 += np.random.normal(0, 5, shape)
    t2[~brain_mask] = 0

    # FLAIR: Suppresses CSF, highlights non-enhancing hyperintensities and vasogenic edema
    flair = base_brain.copy()
    flair[edema_mask] += 95.0 # Very bright edema
    flair[tc_mask] += 40.0
    flair += np.random.normal(0, 5, shape)
    flair[~brain_mask] = 0

    meta = {
        "spacing": spacing,
        "origin": (0.0, 0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }

    # Save to disk
    t1_path = os.path.join(output_dir, f"{case_id}_t1.nii.gz")
    t1ce_path = os.path.join(output_dir, f"{case_id}_t1ce.nii.gz")
    t2_path = os.path.join(output_dir, f"{case_id}_t2.nii.gz")
    flair_path = os.path.join(output_dir, f"{case_id}_flair.nii.gz")
    seg_path = os.path.join(output_dir, f"{case_id}_seg.nii.gz")

    DicomSeriesReader.save_nifti_volume(t1, t1_path, meta, is_label=False)
    DicomSeriesReader.save_nifti_volume(t1ce, t1ce_path, meta, is_label=False)
    DicomSeriesReader.save_nifti_volume(t2, t2_path, meta, is_label=False)
    DicomSeriesReader.save_nifti_volume(flair, flair_path, meta, is_label=False)
    DicomSeriesReader.save_nifti_volume(seg, seg_path, meta, is_label=True)

    metadata_path = os.path.join(output_dir, f"{case_id}_meta.json")
    case_meta = {
        "case_id": case_id,
        "tumor_type": tumor_type,
        "classification_label": 1 if "Glioblastoma" in tumor_type else (0 if "Glioma" in tumor_type else 2),
        "shape": list(shape),
        "spacing": list(spacing),
        "paths": {
            "t1": t1_path,
            "t1ce": t1ce_path,
            "t2": t2_path,
            "flair": flair_path,
            "seg": seg_path
        }
    }
    with open(metadata_path, "w") as f:
        json.dump(case_meta, f, indent=2)

    return case_meta["paths"]
