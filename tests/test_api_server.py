"""
Test client for FastAPI DICOMweb & MONAI Label AI service.
Direct endpoint integration tests verifying QIDO-RS, WADO-RS, STOW-RS,
MONAI inference, active learning, and MPR slice rendering.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.service.app import (
    initialize_system,
    health_check,
    qido_search_studies,
    wado_retrieve_instances,
    stow_store_instances,
    run_monai_inference,
    get_next_active_learning_candidate,
    submit_active_learning_feedback,
    render_slice_image,
    InferenceRequest,
    FeedbackRequest
)

def test_all_endpoints():
    print("[*] Initializing system and database...")
    initialize_system()

    # 1. Health check
    h = health_check()
    print(f"[*] Health Check: {h}")
    assert h["status"] == "healthy"
    assert h["registered_cases"] >= 3

    # 2. QIDO-RS
    studies = qido_search_studies()
    print(f"[*] QIDO-RS returned {len(studies)} studies")
    assert len(studies) >= 3
    assert "00100020" in studies[0] # PatientID tag

    # 3. WADO-RS
    instances = wado_retrieve_instances("study_123", "series_456")
    print(f"[*] WADO-RS metadata: {instances['modality']}, instances: {instances['num_instances']}")
    assert instances["num_instances"] == 128

    # 4. STOW-RS
    stow = stow_store_instances({"sop_instance_uid": "2.25.test_123"})
    print(f"[*] STOW-RS response: {stow['status']}")
    assert stow["status"] == "stored"

    # 5. MONAI Label Inference with Uncertainty & DICOM SEG export
    infer_req = InferenceRequest(case_id="BraTS_2026_001", mc_dropout_passes=5)
    infer_res = run_monai_inference("swin_unetr", infer_req)
    print(f"[*] MONAI Inference Verdict: {infer_res['classification']['predicted_verdict']} (Confidence: {infer_res['classification']['confidence']})")
    print(f"    - Subregion Volumetrics: {infer_res['volumetric_metrics_mm3']}")
    print(f"    - Uncertainty: {infer_res['uncertainty_metrics']}")
    assert infer_res["status"] == "success"
    assert "whole_tumor_wt" in infer_res["volumetric_metrics_mm3"]

    # 6. Active Learning Priority Queue
    al_candidate = get_next_active_learning_candidate()
    print(f"[*] Active Learning Top Priority: {al_candidate['status']}")
    assert al_candidate["status"] == "candidate_found"

    # 7. Clinician Feedback Submission
    fb_req = FeedbackRequest(
        case_id="BraTS_2026_001",
        clinician_id="DR_NEURO_ONCOLOGIST",
        corrections_applied=False,
        notes="Validated tumor margins"
    )
    fb_res = submit_active_learning_feedback(fb_req)
    print(f"[*] Active Learning Feedback: {fb_res['status']}")
    assert fb_res["status"] == "accepted"

    # 8. Multi-Planar Slice Rendering
    slice_res = render_slice_image("BraTS_2026_001", "t1ce", "axial", 32, show_overlay=True)
    print(f"[*] MPR Slice Render: bytes={len(slice_res.body)}, type={slice_res.media_type}")
    assert len(slice_res.body) > 1000
    assert slice_res.media_type == "image/png"

    print("\n[OK] All API & Integration Tests Passed Successfully!")

if __name__ == "__main__":
    test_all_endpoints()
