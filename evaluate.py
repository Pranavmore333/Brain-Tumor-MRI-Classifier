import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from mri_dataset import MRIImageDataset
from models.cnn_model import CLASS_NAMES, get_custom_cnn


def evaluate(
    data_dir: str = "dataset/Testing",
    checkpoint_path: str = "checkpoints/custom_cnn_best.pth",
) -> None:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    dataset = MRIImageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    model = get_custom_cnn(num_classes=len(CLASS_NAMES)).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    expected = []
    predicted = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            expected.extend(labels.tolist())
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())

    print(f"Held-out test images: {len(dataset)}")
    print(classification_report(
        expected, predicted, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, zero_division=0
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(expected, predicted, labels=list(range(len(CLASS_NAMES)))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained CNN on a separate labeled test folder."
    )
    parser.add_argument("--data-dir", default="dataset/Testing")
    parser.add_argument(
        "--checkpoint", default="checkpoints/custom_cnn_best.pth"
    )
    args = parser.parse_args()
    evaluate(args.data_dir, args.checkpoint)


if __name__ == "__main__":
    main()
