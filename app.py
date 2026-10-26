import io
import os
import pickle
import threading
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

import database
from models.cnn_model import CLASS_NAMES, get_custom_cnn
from models.gradcam import GradCAM


IMAGE_SIZE = 224
MAX_IMAGE_SIDE = 4096
MODEL_NAME = "Four-block CNN"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    if not os.path.isfile(checkpoint_path):
        return None, None
    try:
        model = get_custom_cnn(num_classes=len(CLASS_NAMES)).to(DEVICE)
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, None
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as error:
        return None, str(error)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = (pixels - mean) / std
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(DEVICE)


def _read_upload(upload) -> Image.Image:
    if not upload or not upload.filename:
        raise ValueError("Choose a PNG or JPG image to analyze.")
    if os.path.splitext(upload.filename)[1].lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Only PNG and JPG images are supported.")

    content = upload.read()
    try:
        with Image.open(io.BytesIO(content)) as check:
            check.verify()
        with Image.open(io.BytesIO(content)) as image:
            if image.width < 32 or image.height < 32:
                raise ValueError("The image must be at least 32 pixels wide and tall.")
            if image.width > MAX_IMAGE_SIDE or image.height > MAX_IMAGE_SIDE:
                raise ValueError("The image must not exceed 4096 pixels on either side.")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The selected file is not a readable PNG or JPG image.") from error


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    application = Flask(__name__)
    application.config.from_mapping(
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        UPLOAD_FOLDER=os.path.join(application.root_path, "static", "uploads"),
        DATABASE_PATH=database.DB_PATH,
        MODEL_PATH=os.path.join(
            application.root_path, "checkpoints", "custom_cnn_best.pth"
        ),
    )
    if test_config:
        application.config.update(test_config)

    os.makedirs(application.config["UPLOAD_FOLDER"], exist_ok=True)
    database.init_db(application.config["DATABASE_PATH"])
    model, model_error = load_model(application.config["MODEL_PATH"])
    if model_error:
        application.logger.error("Could not load model checkpoint: %s", model_error)
    elif model is None:
        application.logger.warning(
            "No trained CNN checkpoint found. Train the model before making predictions."
        )

    application.extensions["mri_model"] = model
    application.extensions["mri_model_error"] = model_error
    application.extensions["mri_gradcam"] = (
        GradCAM(model, model.conv4) if model is not None else None
    )
    inference_lock = threading.Lock()

    @application.errorhandler(413)
    def file_too_large(_error):
        return jsonify({"error": "Image exceeds the 8 MB upload limit."}), 413

    @application.get("/")
    def index():
        model_ready = application.extensions["mri_model"] is not None
        model_error = application.extensions["mri_model_error"]
        return render_template(
            "index.html",
            active_page="analyze",
            model_ready=model_ready,
            model_error=bool(model_error),
            model_status=(
                "Ready to analyze" if model_ready else
                "Checkpoint could not be loaded" if model_error else
                "Train the model to enable analysis"
            ),
        )

    @application.get("/history")
    def history():
        search_query = request.args.get("q", "").strip()
        filter_class = request.args.get("class", "all").strip()
        scans = database.get_all_scans(
            search_query, filter_class, application.config["DATABASE_PATH"]
        )
        stats = database.get_stats(application.config["DATABASE_PATH"])
        return render_template(
            "history.html",
            active_page="history",
            scans=scans,
            stats=stats,
            search_query=search_query,
            filter_class=filter_class,
            class_names=CLASS_NAMES,
        )

    @application.get("/about")
    def about():
        return render_template("about.html", active_page="about")

    @application.post("/api/predict")
    def predict():
        model = application.extensions["mri_model"]
        if model is None:
            return jsonify({
                "error": (
                    "A trained checkpoint is required. Follow the training steps "
                    "on the About page, then restart the app."
                )
            }), 503

        try:
            image = _read_upload(request.files.get("image"))
            case_label = request.form.get("case_label", "").strip()
            notes = request.form.get("notes", "").strip()
            if len(case_label) > 64:
                raise ValueError("Case label must be 64 characters or fewer.")
            if len(notes) > 1000:
                raise ValueError("Review note must be 1000 characters or fewer.")
            if not case_label:
                case_label = f"case-{uuid.uuid4().hex[:8]}"

            input_tensor = preprocess_image(image)
            gradcam = application.extensions["mri_gradcam"]
            with inference_lock:
                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
                predicted_index = int(np.argmax(probabilities))
                heatmap = gradcam.generate(input_tensor, class_idx=predicted_index)
                overlay = gradcam.overlay_on_image(image, heatmap, alpha=0.45)

            file_id = uuid.uuid4().hex
            original_name = f"{file_id}.png"
            gradcam_name = f"{file_id}_gradcam.png"
            upload_folder = application.config["UPLOAD_FOLDER"]
            probability_results = {
                CLASS_NAMES[index]: round(float(value) * 100, 1)
                for index, value in enumerate(probabilities)
            }
            confidence = float(probabilities[predicted_index]) * 100
            try:
                image.save(os.path.join(upload_folder, original_name), format="PNG")
                overlay.save(os.path.join(upload_folder, gradcam_name), format="PNG")
                case_id = database.insert_scan(
                    case_label=case_label,
                    original_image=original_name,
                    gradcam_image=gradcam_name,
                    predicted_class=CLASS_NAMES[predicted_index],
                    confidence=confidence,
                    model_used=MODEL_NAME,
                    probabilities=probability_results,
                    notes=notes,
                    db_path=application.config["DATABASE_PATH"],
                )
            except Exception:
                for filename in (original_name, gradcam_name):
                    path = os.path.join(upload_folder, filename)
                    if os.path.isfile(path):
                        os.remove(path)
                raise

            return jsonify({
                "success": True,
                "case_id": case_id,
                "case_label": case_label,
                "predicted_class": CLASS_NAMES[predicted_index],
                "confidence": round(confidence, 1),
                "model_used": MODEL_NAME,
                "probabilities": probability_results,
                "original_image_url": f"/static/uploads/{original_name}",
                "gradcam_image_url": f"/static/uploads/{gradcam_name}",
                "history_url": "/history",
            })
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Exception:
            application.logger.exception("MRI image analysis failed")
            return jsonify({"error": "Analysis failed. Check the server log for details."}), 500

    @application.get("/api/cases")
    def list_cases():
        scans = database.get_all_scans(db_path=application.config["DATABASE_PATH"])
        return jsonify({"count": len(scans), "cases": scans})

    @application.post("/api/cases/<int:case_id>/review")
    def review_case(case_id: int):
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Send a JSON object with the review fields."}), 400
        reviewed_class = data.get("reviewed_class") or None
        review_note = data.get("review_note", "")
        if not isinstance(review_note, str):
            return jsonify({"error": "Review note must be text."}), 400
        review_note = review_note.strip()
        if reviewed_class is not None and reviewed_class not in CLASS_NAMES:
            return jsonify({"error": "Choose one of the listed review labels."}), 400
        if len(review_note) > 1000:
            return jsonify({"error": "Review note must be 1000 characters or fewer."}), 400
        updated = database.update_review(
            case_id,
            reviewed_class,
            review_note,
            application.config["DATABASE_PATH"],
        )
        if not updated:
            return jsonify({"error": "Case not found."}), 404
        return jsonify({"success": True})

    @application.delete("/api/cases/<int:case_id>")
    def remove_case(case_id: int):
        scan = database.get_scan_by_id(case_id, application.config["DATABASE_PATH"])
        if scan is None:
            return jsonify({"error": "Case not found."}), 404
        if not database.delete_scan(case_id, application.config["DATABASE_PATH"]):
            return jsonify({"error": "Case could not be deleted."}), 500

        upload_folder = os.path.realpath(application.config["UPLOAD_FOLDER"])
        for filename in (scan["original_image"], scan["gradcam_image"]):
            if filename:
                path = os.path.realpath(os.path.join(upload_folder, os.path.basename(filename)))
                if os.path.dirname(path) == upload_folder and os.path.isfile(path):
                    os.remove(path)
        return jsonify({"success": True})

    return application


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(host=host, port=port, debug=False)
