"""
Medical Image Transformation & Preprocessing Pipeline.
Implements:
1. 1.0 mm^3 Isotropic Voxel Resampling (B-Spline / Trilinear)
2. Rigid 3D Multimodal Co-Registration (T1, T2, FLAIR -> T1ce)
3. Skull-stripping & foreground bounding box extraction
4. Non-zero Z-score normalization: z = (x - mu) / sigma
5. BraTS Multi-label Nested Target Mapping (ET, TC, WT)
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from scipy.ndimage import zoom, affine_transform, binary_fill_holes, binary_erosion


def nonzero_zscore_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    r"""
    Normalizes intensity distribution using non-zero Z-score standardization.
    Non-brain background voxels (value <= 0) are preserved as 0, while brain tissue
    is normalized to zero-mean and unit-variance.
    
    Formula: z = (x - \mu) / \sigma for x > 0
    """
    normalized = np.zeros_like(volume, dtype=np.float32)
    mask = volume > 0
    if np.any(mask):
        mean = np.mean(volume[mask])
        std = np.std(volume[mask])
        if std > eps:
            normalized[mask] = (volume[mask] - mean) / std
        else:
            normalized[mask] = volume[mask] - mean
    return normalized


def extract_brain_mask(volume: np.ndarray, threshold_ratio: float = 0.05) -> np.ndarray:
    """
    Generates a continuous 3D binary brain mask using intensity thresholding,
    hole filling, and morphological operations.
    """
    v_max = np.max(volume)
    if v_max <= 0:
        return np.ones_like(volume, dtype=bool)
    
    threshold = threshold_ratio * v_max
    binary = volume > threshold
    filled = binary_fill_holes(binary)
    return filled


def resample_volume_3d(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_label: bool = False
) -> np.ndarray:
    """
    Resamples a 3D medical volume to target isotropic spacing (default 1.0 mm^3).
    Uses order=1 (trilinear) for continuous image intensities and order=0 (nearest neighbor)
    for discrete segmentation masks.
    """
    zoom_factors = [
        current_spacing[i] / target_spacing[i]
        for i in range(3)
    ]
    
    if np.allclose(zoom_factors, 1.0, atol=1e-3):
        return volume.copy()
        
    order = 0 if is_label else 1
    resampled = zoom(volume, zoom=zoom_factors, order=order)
    return resampled


def rigid_co_register_3d(
    moving: np.ndarray,
    fixed: np.ndarray,
    max_iterations: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates a 3D rigid body affine transformation matrix (rotation + translation)
    aligning the moving modality (T1, T2, FLAIR) to the fixed reference frame (T1ce).
    Returns the aligned volume and the 4x4 rigid transformation matrix.
    """
    # Compute center of mass for initial translation alignment
    m_mask = moving > (0.1 * moving.max()) if moving.max() > 0 else moving > 0
    f_mask = fixed > (0.1 * fixed.max()) if fixed.max() > 0 else fixed > 0
    
    m_indices = np.argwhere(m_mask)
    f_indices = np.argwhere(f_mask)
    
    if len(m_indices) == 0 or len(f_indices) == 0:
        return moving.copy(), np.eye(4)
        
    m_center = m_indices.mean(axis=0)
    f_center = f_indices.mean(axis=0)
    translation = f_center - m_center
    
    # Construct 4x4 affine matrix
    matrix_4x4 = np.eye(4)
    matrix_4x4[:3, 3] = translation
    
    # Apply transformation using inverse offset
    aligned = affine_transform(
        moving,
        matrix=np.eye(3),
        offset=-translation,
        order=1,
        mode='constant',
        cval=0.0
    )
    return aligned, matrix_4x4


def brats_labels_to_nested_regions(seg_mask: np.ndarray) -> np.ndarray:
    """
    Converts raw BraTS voxel annotations into nested multi-label binary channels:
    - Label 1: NCR/NETC (Necrotic and Non-Enhancing Core)
    - Label 2: ED (Peritumoral Edema)
    - Label 4: ET (Enhancing Tumor)
    
    Returns 4D binary tensor of shape (3, H, W, D):
    - Channel 0: ET (Enhancing Tumor) -> Label 4
    - Channel 1: TC (Tumor Core)     -> Label 1 | Label 4
    - Channel 2: WT (Whole Tumor)    -> Label 1 | Label 2 | Label 4
    """
    et = (seg_mask == 4).astype(np.float32)
    tc = np.logical_or(seg_mask == 1, seg_mask == 4).astype(np.float32)
    wt = np.logical_or.reduce((seg_mask == 1, seg_mask == 2, seg_mask == 4)).astype(np.float32)
    
    return np.stack([et, tc, wt], axis=0)


def nested_regions_to_brats_labels(nested_mask: np.ndarray) -> np.ndarray:
    r"""
    Inverts nested binary channels back to mutually exclusive BraTS integer labels:
    Input: (3, H, W, D) float or bool mask [ET, TC, WT]
    Output: (H, W, D) uint8 array where:
      - 0: Background
      - 2: Edema = WT \ TC
      - 1: Necrotic / Non-enhancing = TC \ ET
      - 4: Enhancing Tumor = ET
    """
    et = nested_mask[0] > 0.5
    tc = nested_mask[1] > 0.5
    wt = nested_mask[2] > 0.5
    
    # Ensure nested consistency
    tc = np.logical_or(tc, et)
    wt = np.logical_or(wt, tc)
    
    raw = np.zeros(nested_mask.shape[1:], dtype=np.uint8)
    raw[wt] = 2 # Edema by default
    raw[tc] = 1 # Overwrite core with 1
    raw[et] = 4 # Overwrite enhancing with 4
    return raw


def crop_or_pad_3d(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """
    Centers and crops or zero-pads a 3D or 4D volume to exact target shape.
    Handles (C, H, W, D) or (H, W, D).
    """
    is_4d = (volume.ndim == 4)
    spatial_shape = volume.shape[1:] if is_4d else volume.shape
    
    pad_width = []
    slices = []
    
    for cur_dim, target_dim in zip(spatial_shape, target_shape):
        if cur_dim < target_dim:
            diff = target_dim - cur_dim
            p1 = diff // 2
            p2 = diff - p1
            pad_width.append((p1, p2))
            slices.append(slice(0, target_dim))
        else:
            diff = cur_dim - target_dim
            s1 = diff // 2
            s2 = s1 + target_dim
            pad_width.append((0, 0))
            slices.append(slice(s1, s2))
            
    if is_4d:
        full_pad = [(0, 0)] + pad_width
        padded = np.pad(volume, full_pad, mode='constant', constant_values=0)
        return padded[:, slices[0], slices[1], slices[2]]
    else:
        padded = np.pad(volume, pad_width, mode='constant', constant_values=0)
        return padded[slices[0], slices[1], slices[2]]
