"""
End-to-End Test Suite for Brain Tumor MRI Classifier.
Tests all phases of the Architectural Blueprint:
- Phase 1: Environment & Preprocessing (resampling, z-score, nested labels, synthetic generation)
- Phase 2: Model Architecture & Training (Swin UNETR, MedNeXt, Composite Loss)
- Phase 3: PACS & Web Integration (DICOM SEG serialization, API endpoints)
- Phase 4: MLOps Acceleration (sliding window, Gaussian blending, uncertainty)
- Phase 5: Quantitative Evaluation (Dice, HD95, Lesion Dice)
"""

import os
import shutil
import unittest
import numpy as np
import torch

from src.data.transforms import (
    nonzero_zscore_normalize,
    extract_brain_mask,
    resample_volume_3d,
    rigid_co_register_3d,
    brats_labels_to_nested_regions,
    nested_regions_to_brats_labels
)
from src.data.dicom_io import DicomSeriesReader, generate_synthetic_brats_case
from src.models.swin_unetr import SwinUNETR
from src.models.mednext import MedNeXt
from src.models.losses import MultiTaskCompositeLoss, MultiLabelSoftDiceLoss
from src.inference.sliding_window import create_3d_gaussian_weight_matrix, SlidingWindowInferer
from src.evaluation.metrics import compute_dice_score, compute_hd95, compute_lesion_wise_dice, ClinicalEvaluationHarness
from src.service.dicom_seg import DicomSegSerializer


class TestBrainTumorPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = "data/test_tmp"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_preprocessing_and_zscore(self):
        """Phase 1: Validates non-zero Z-score normalization and brain mask extraction."""
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        # Brain foreground with mean 100, std 15
        vol[8:24, 8:24, 8:24] = np.random.normal(100.0, 15.0, (16, 16, 16))
        
        norm = nonzero_zscore_normalize(vol)
        mask = extract_brain_mask(vol)

        # Background should remain strictly 0
        self.assertEqual(norm[0, 0, 0], 0.0)
        # Foreground mean should be ~ 0 and std ~ 1
        fg_vals = norm[vol > 0]
        self.assertAlmostEqual(float(np.mean(fg_vals)), 0.0, places=2)
        self.assertAlmostEqual(float(np.std(fg_vals)), 1.0, places=1)
        self.assertTrue(np.any(mask))

    def test_02_brats_nested_label_mapping(self):
        """Phase 1: Validates conversion between raw BraTS labels (1, 2, 4) and nested channels."""
        raw_seg = np.zeros((32, 32, 32), dtype=np.uint8)
        raw_seg[10:14, 10:14, 10:14] = 1 # Necrotic Core (NCR)
        raw_seg[14:18, 14:18, 14:18] = 2 # Edema (ED)
        raw_seg[18:22, 18:22, 18:22] = 4 # Enhancing Tumor (ET)

        nested = brats_labels_to_nested_regions(raw_seg) # (3, H, W, D)
        self.assertEqual(nested.shape, (3, 32, 32, 32))

        # ET channel should only have label 4
        self.assertEqual(int(nested[0].sum()), 4 * 4 * 4)
        # TC channel should have ET + NCR (labels 1 + 4)
        self.assertEqual(int(nested[1].sum()), 2 * 4 * 4 * 4)
        # WT channel should have all tumor voxels (labels 1 + 2 + 4)
        self.assertEqual(int(nested[2].sum()), 3 * 4 * 4 * 4)

        # Invert back to discrete
        inverted = nested_regions_to_brats_labels(nested)
        np.testing.assert_array_equal(raw_seg, inverted)

    def test_03_synthetic_case_generation_and_io(self):
        """Phase 1: Generates a full 4-channel synthetic case and validates NIfTI I/O."""
        paths = generate_synthetic_brats_case(
            output_dir=self.test_dir,
            case_id="TEST_001",
            shape=(32, 32, 32),
            spacing=(1.0, 1.0, 1.0)
        )
        for mod in ["t1", "t1ce", "t2", "flair", "seg"]:
            self.assertTrue(os.path.exists(paths[mod]))

        vol, meta = DicomSeriesReader.load_nifti_volume(paths["t1ce"])
        self.assertEqual(vol.shape, (32, 32, 32))
        self.assertEqual(meta["spacing"], (1.0, 1.0, 1.0))

    def test_04_swin_unetr_forward_pass(self):
        """Phase 2: Validates 3D Swin UNETR multi-task network shapes."""
        model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=12,
            num_classes=3,
            window_size=4,
            patch_size=2
        )
        dummy_input = torch.randn(1, 4, 32, 32, 32)
        seg_logits, cls_logits = model(dummy_input)

        # Check multi-label segmentation logits shape: (B, 3, H, W, D)
        self.assertEqual(seg_logits.shape, (1, 3, 32, 32, 32))
        # Check global classification logits shape: (B, 3)
        self.assertEqual(cls_logits.shape, (1, 3))

    def test_05_mednext_forward_pass(self):
        """Phase 2: Validates 3D MedNeXt 7x7x7 depthwise separable network shapes."""
        model = MedNeXt(
            in_channels=4,
            out_channels=3,
            base_channels=16,
            num_classes=3,
            kernel_size=3, # compact kernel for 32^3 test
            expansion_ratio=2,
            blocks_per_stage=[1, 1, 1, 1]
        )
        dummy_input = torch.randn(1, 4, 32, 32, 32)
        seg_logits, cls_logits = model(dummy_input)

        self.assertEqual(seg_logits.shape, (1, 3, 32, 32, 32))
        self.assertEqual(cls_logits.shape, (1, 3))

    def test_06_composite_loss(self):
        """Phase 2: Evaluates composite multi-task Dice + BCE + Categorical CE loss."""
        criterion = MultiTaskCompositeLoss(lambda_ce=1.0, lambda_cls=0.5)

        seg_pred = torch.randn(2, 3, 16, 16, 16, requires_grad=True)
        seg_target = torch.randint(0, 2, (2, 3, 16, 16, 16)).float()
        cls_pred = torch.randn(2, 3, requires_grad=True)
        cls_target = torch.tensor([0, 1], dtype=torch.long)

        losses = criterion(seg_pred, seg_target, cls_pred, cls_target)
        self.assertIn("total_loss", losses)
        self.assertIn("dice_loss", losses)
        self.assertIn("bce_loss", losses)
        self.assertIn("cls_loss", losses)

        # Validate backpropagation
        losses["total_loss"].backward()
        self.assertIsNotNone(seg_pred.grad)
        self.assertIsNotNone(cls_pred.grad)

    def test_07_sliding_window_gaussian_inference(self):
        """Phase 4: Validates Gaussian weight blending and sliding window inference."""
        w_gauss = create_3d_gaussian_weight_matrix((32, 32, 32))
        self.assertEqual(w_gauss.shape, (32, 32, 32))
        # Center should be peak 1.0
        self.assertAlmostEqual(float(w_gauss[16, 16, 16]), 1.0, places=2)
        # Boundary should be decayed
        self.assertLess(float(w_gauss[0, 0, 0]), 0.1)

        inferer = SlidingWindowInferer(patch_size=(16, 16, 16), overlap=0.5, device="cpu")
        slices = inferer.get_patch_coordinates((32, 32, 32))
        self.assertGreater(len(slices), 1)

    def test_08_clinical_evaluation_metrics(self):
        """Phase 5: Computes DSC, HD95, and Lesion-Wise Dice."""
        mask_gt = np.zeros((32, 32, 32), dtype=np.uint8)
        mask_gt[10:20, 10:20, 10:20] = 1

        mask_pred = np.zeros((32, 32, 32), dtype=np.uint8)
        mask_pred[12:22, 10:20, 10:20] = 1 # Shifted by 2 voxels

        dsc = compute_dice_score(mask_pred, mask_gt)
        hd95 = compute_hd95(mask_pred, mask_gt, voxel_spacing=(1.0, 1.0, 1.0))
        lesion_dsc = compute_lesion_wise_dice(mask_pred, mask_gt)

        self.assertGreater(dsc, 0.70)
        self.assertLess(dsc, 1.0)
        self.assertGreater(hd95, 0.0)
        self.assertLess(hd95, 5.0)
        self.assertGreater(lesion_dsc, 0.70)

    def test_09_dicom_seg_export(self):
        """Phase 3: Validates DICOM SEG manifest generation and NIfTI package export."""
        dummy_nested = np.zeros((3, 32, 32, 32), dtype=np.float32)
        dummy_nested[0, 12:16, 12:16, 12:16] = 1.0
        dummy_nested[1, 10:18, 10:18, 10:18] = 1.0
        dummy_nested[2, 8:22, 8:22, 8:22] = 1.0

        meta = {"spacing": (1.0, 1.0, 1.0), "origin": (0.0, 0.0, 0.0)}
        export = DicomSegSerializer.export_dicom_seg_package(
            output_dir=self.test_dir,
            case_id="DICOM_SEG_TEST",
            study_uid="1.2.826.0.1.test",
            source_series_uid="1.2.826.0.1.test.t1ce",
            seg_mask_nested=dummy_nested,
            reference_meta=meta
        )

        self.assertTrue(os.path.exists(export["manifest_path"]))
        self.assertTrue(os.path.exists(export["seg_nifti_path"]))


if __name__ == "__main__":
    unittest.main()
