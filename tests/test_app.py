import io
import os
import tempfile
import unittest

import torch
from PIL import Image

import database
from app import create_app
from models.cnn_model import CLASS_NAMES, get_custom_cnn


class SliceLabAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "cases.sqlite")
        self.upload_dir = os.path.join(self.temp_dir.name, "uploads")
        os.makedirs(self.upload_dir)
        self.app = create_app({
            "TESTING": True,
            "DATABASE_PATH": self.db_path,
            "UPLOAD_FOLDER": self.upload_dir,
            "MODEL_PATH": os.path.join(self.temp_dir.name, "missing.pth"),
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_homepage_explains_checkpoint_is_required(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Train the model to enable analysis", response.data)

    def test_prediction_is_unavailable_without_a_trained_checkpoint(self):
        response = self.client.post("/api/predict")
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"trained checkpoint is required", response.data)

    def test_review_feedback_round_trip(self):
        case_id = database.insert_scan(
            case_label="study-01",
            original_image="input.png",
            gradcam_image="overlay.png",
            predicted_class=CLASS_NAMES[0],
            confidence=80.0,
            model_used="Test CNN",
            probabilities={CLASS_NAMES[0]: 80.0},
            notes="",
            db_path=self.db_path,
        )
        response = self.client.post(
            f"/api/cases/{case_id}/review",
            json={"reviewed_class": CLASS_NAMES[1], "review_note": "Review example"},
        )
        self.assertEqual(response.status_code, 200)
        cases = self.client.get("/api/cases").get_json()["cases"]
        self.assertEqual(cases[0]["reviewed_class"], CLASS_NAMES[1])
        self.assertEqual(cases[0]["review_note"], "Review example")
        self.assertNotIn("patient_name", cases[0])

    def test_uploaded_image_runs_through_model_gradcam_and_database(self):
        checkpoint = os.path.join(self.temp_dir.name, "test-only-checkpoint.pth")
        torch.save(get_custom_cnn().state_dict(), checkpoint)
        app = create_app({
            "TESTING": True,
            "DATABASE_PATH": self.db_path,
            "UPLOAD_FOLDER": self.upload_dir,
            "MODEL_PATH": checkpoint,
        })
        image_bytes = io.BytesIO()
        Image.new("RGB", (64, 64), color="gray").save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = app.test_client().post(
            "/api/predict",
            data={
                "image": (image_bytes, "slice.png"),
                "case_label": "test-case",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        result = response.get_json()
        self.assertEqual(result["case_label"], "test-case")
        self.assertIn(result["predicted_class"], CLASS_NAMES)
        self.assertEqual(set(result["probabilities"]), set(CLASS_NAMES))
        cases = database.get_all_scans(db_path=self.db_path)
        self.assertEqual(len(cases), 1)
        self.assertTrue(os.path.isfile(os.path.join(self.upload_dir, cases[0]["original_image"])))
        self.assertTrue(os.path.isfile(os.path.join(self.upload_dir, cases[0]["gradcam_image"])))


if __name__ == "__main__":
    unittest.main()
