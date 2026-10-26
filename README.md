# SliceLab: brain MRI learning project

SliceLab is a small, end-to-end deep-learning project intended for an individual undergraduate. It is deliberately one model and one workflow:

1. Train a custom four-block CNN with PyTorch.
2. Upload one 2D MRI slice in a Flask web app.
3. View the predicted class, class scores, and a Grad-CAM overlay.
4. Keep the result in a local SQLite case log and optionally add a human review label.

The review log is the project's distinguishing feature: it lets a student inspect where the model and a reviewer disagree instead of presenting a prediction as a finished diagnosis. Review labels are not automatically treated as verified ground truth.

## What is included

- `app.py`: Flask pages and JSON endpoints.
- `models/cnn_model.py`: the single CNN architecture.
- `models/gradcam.py`: Grad-CAM visualization for that CNN.
- `mri_dataset.py`: image loading and a class-stratified validation split.
- `train.py`: training and best-validation checkpoint saving.
- `evaluate.py`: classification report and confusion matrix on a separate test folder.
- `database.py`: SQLite persistence for predictions and optional review notes.
- `templates/`, `static/`: responsive interface and browser interactions.

No dataset, fabricated sample images, pre-trained project weights, or made-up benchmark results are included. The app disables inference until a real checkpoint has been trained.

## Setup

Python 3.12 is recommended (use a version with a compatible PyTorch wheel). In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch's default package index does not provide a build suitable for your computer, install the CPU or CUDA build recommended by the [PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements.

## Prepare the data

Use a dataset you are allowed to use. The common four-folder brain-tumor MRI image dataset layout is supported:

```text
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Folder names are case-insensitive. `no tumor` and `no_tumor` are also accepted in place of `notumor`. Put PNG, JPG, or JPEG images directly in each class folder. Do not upload patient-identifiable images or information to a shared environment.

For a stronger experiment, split by patient rather than by individual image. Nearby slices from the same patient can be very similar; allowing them into both training and validation/test sets can make the reported score misleading. Keep the test set separate until final evaluation.

## Train, evaluate, run

Run from the repository root:

```powershell
python train.py --data-dir dataset/Training --epochs 10
python evaluate.py --data-dir dataset/Testing
python app.py
```

Training stratifies `dataset/Training` into an 80/20 train/validation split, applies a small horizontal-flip augmentation to training images, and writes the best-validation model to `checkpoints/custom_cnn_best.pth`. Evaluation reports metrics from `dataset/Testing`; those values are only meaningful for the data you actually supplied. The web app uses the same checkpoint. Open <http://127.0.0.1:5000>.

The application accepts one 2D PNG/JPG image (up to 8 MB). It does not read DICOM or NIfTI volumes, perform tumor segmentation, connect to a hospital/PACS system, or provide a clinical triage score. SQLite and uploaded images are stored locally in `instance/` and `static/uploads/`.

## Who could benefit, and realistic next steps

- **Students and instructors:** a compact example of CNN training, image preprocessing, explainability, a Flask API, SQL persistence, and error review.
- **Junior AI/ML applicants:** a portfolio demonstration of the full path from data to model evaluation and a usable interface. Present real test results, limitations, and your own contributions honestly.
- **Research or teaching groups:** a possible offline teaching aid or starting point for studying model errors, provided the dataset is licensed appropriately and the work is independently validated.

The realistic short-term income route is using the project to support job applications, internships, tutoring, or workshops. Selling a clinical prediction service is not a responsible next step for this prototype: it would require representative external validation, patient-level evidence, privacy/security controls, clinical governance, and applicable regulatory approval.

## Safety and limitations

This is coursework software, not a medical device. Class scores are not calibrated diagnostic probabilities. Grad-CAM shows image regions that influenced the model; it does not prove that those regions contain a tumor or explain a medical decision. Never use the application to diagnose, triage, or treat a person. Use de-identified data and consult dataset terms before training or sharing results.
