import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from mri_dataset import MRIImageDataset, split_training_data
from models.cnn_model import CLASS_NAMES, get_custom_cnn


def train_model(
    data_dir: str = "dataset/Training",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    checkpoint_path: str = "checkpoints/custom_cnn_best.pth",
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmented_data = MRIImageDataset(data_dir, augment=True)
    validation_data = MRIImageDataset(data_dir, augment=False)
    train_indices, validation_indices = split_training_data(augmented_data)
    train_loader = DataLoader(
        Subset(augmented_data, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(validation_data, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = get_custom_cnn(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    best_accuracy = -1.0
    best_state = None

    print(
        f"Training one custom CNN on {device}: "
        f"{len(train_indices)} training and {len(validation_indices)} validation images."
    )
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = model(images.to(device))
                correct += int((outputs.argmax(dim=1).cpu() == labels).sum())
        accuracy = correct / len(validation_indices)
        mean_loss = train_loss / len(train_indices)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} - "
            f"train loss: {mean_loss:.4f}, validation accuracy: {accuracy:.1%}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    torch.save(best_state, checkpoint_path)
    print(f"Best validation accuracy: {best_accuracy:.1%}")
    print(f"Saved checkpoint: {checkpoint_path}")
    return best_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CNN on a labeled MRI image dataset."
    )
    parser.add_argument("--data-dir", default="dataset/Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--checkpoint", default="checkpoints/custom_cnn_best.pth"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive.")
    train_model(
        args.data_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
