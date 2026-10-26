# Brain Tumor MRI Classifier

A Flask application for classifying individual 2D brain MRI images into four
categories: glioma, meningioma, no tumor, and pituitary. The classifier is a
custom PyTorch CNN. Results include class scores, a Grad-CAM overlay, and a
record in a local SQLite database.

This repository contains the application and training code, not an MRI dataset
or trained model. Train the model with your own permitted data before using the
analysis page.

## Requirements

Use Python 3.12 and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a compatible CPU or CUDA build of PyTorch first if the default package
index does not provide one for your system. The
[PyTorch installation guide](https://pytorch.org/get-started/locally/) provides
the appropriate command.

## Data layout

Place labeled PNG or JPG images in these folders:

```text
dataset/
    Training/
        glioma/
        meningioma/
        notumor/
        pituitary/
    Testing/
        glioma/
        meningioma/
        notumor/
        pituitary/
```

Folder names are case-insensitive. `no_tumor` and `no tumor` can be used instead
of `notumor`. The training folder is split by image into training and
validation sets. If patient identifiers are available, split the data by
patient before training; otherwise, related slices can appear in both sets and
inflate validation scores. Keep the testing folder separate for final
evaluation.

## Train and run

From the repository root:

```powershell
python train.py --data-dir dataset/Training --epochs 10
python evaluate.py --data-dir dataset/Testing
python app.py
```

Training saves the checkpoint with the best validation accuracy to
`checkpoints/custom_cnn_best.pth`. Evaluation prints a classification report
and confusion matrix for the test folder. The Flask app loads that checkpoint
at startup; restart it after retraining. Open
<http://127.0.0.1:5000> in a browser.

The web app accepts one PNG or JPG image up to 8 MB. It does not process DICOM
or NIfTI studies. Predictions and uploaded images are stored locally in
`instance/` and `static/uploads/`. The case log can store an optional reviewer
label and note for later comparison.

## Limitations

This is a research and learning example, not a medical device. Softmax scores
are not calibrated diagnostic probabilities, and Grad-CAM is a visualization
of model activations—not a validated localization. Do not use this application
to diagnose, triage, or treat patients. Use de-identified data and follow its
license. Clinical use would require independent validation, privacy and
security controls, specialist oversight, and applicable regulatory approval.
