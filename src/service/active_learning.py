"""
Active Learning Manager & Priority Review Queue.
Blueprint Specifications:
- Evaluates unlabeled patient scans in the PACS database
- Prioritizes scans with highest Epistemic Uncertainty (Model Uncertainty)
- Serves `/monai/activelearn/next` endpoint for clinical annotation
- Ingests verified clinician refinements for continuous background fine-tuning
"""

import os
import json
import time
from typing import Dict, Any, List, Optional


class ActiveLearningQueue:
    """
    Priority queue managing patient studies ranked by volume uncertainty scores.
    """

    def __init__(self, queue_db_path: str = "data/active_learning_queue.json"):
        self.queue_db_path = queue_db_path
        self.queue: List[Dict[str, Any]] = []
        self._load_queue()

    def _load_queue(self) -> None:
        if os.path.exists(self.queue_db_path):
            try:
                with open(self.queue_db_path, "r") as f:
                    self.queue = json.load(f)
            except Exception:
                self.queue = []
        else:
            self.queue = []

    def _save_queue(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.queue_db_path)), exist_ok=True)
        with open(self.queue_db_path, "w") as f:
            json.dump(self.queue, f, indent=2)

    def register_case(
        self,
        case_id: str,
        study_uid: str,
        volume_uncertainty_score: float,
        preliminary_verdict: str,
        paths: Dict[str, str]
    ) -> None:
        """
        Inserts or updates a study in the review queue with its uncertainty score.
        """
        # Check if already present
        for entry in self.queue:
            if entry["case_id"] == case_id:
                entry["uncertainty_score"] = volume_uncertainty_score
                entry["verdict"] = preliminary_verdict
                entry["timestamp"] = time.time()
                self._save_queue()
                return

        self.queue.append({
            "case_id": case_id,
            "study_uid": study_uid,
            "uncertainty_score": round(volume_uncertainty_score, 4),
            "verdict": preliminary_verdict,
            "paths": paths,
            "status": "pending_review",
            "timestamp": time.time()
        })
        # Sort descending by uncertainty
        self.queue.sort(key=lambda x: x["uncertainty_score"], reverse=True)
        self._save_queue()

    def get_highest_uncertainty_case(self) -> Optional[Dict[str, Any]]:
        """
        Returns the top priority case awaiting clinical refinement.
        """
        pending = [c for c in self.queue if c["status"] == "pending_review"]
        if pending:
            return pending[0]
        return None

    def mark_reviewed(
        self,
        case_id: str,
        clinician_id: str,
        corrections_applied: bool,
        notes: str = ""
    ) -> bool:
        """
        Marks a case as verified by a radiologist and archives it for model retraining.
        """
        for entry in self.queue:
            if entry["case_id"] == case_id:
                entry["status"] = "reviewed"
                entry["reviewed_by"] = clinician_id
                entry["corrections_applied"] = corrections_applied
                entry["review_timestamp"] = time.time()
                entry["notes"] = notes
                self._save_queue()
                return True
        return False
