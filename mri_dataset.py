import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from models.cnn_model import CLASS_NAMES


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_FOLDER_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}


class MRIImageDataset(Dataset):
    def __init__(self, root_dir: str, augment: bool = False, image_size: int = 224):
        self.root_dir = root_dir
        self.augment = augment
        self.image_size = image_size
        self.samples: List[Tuple[str, int]] = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

        class_indices = {name: index for index, name in enumerate(CLASS_NAMES)}
        found_classes = set()
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            label_name = _FOLDER_LABELS.get(
                folder_name.lower().replace(" ", "").replace("_", "").replace("-", "")
            )
            if not os.path.isdir(folder_path) or label_name is None:
                continue
            found_classes.add(label_name)
            for filename in sorted(os.listdir(folder_path)):
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                    self.samples.append(
                        (os.path.join(folder_path, filename), class_indices[label_name])
                    )

        missing_classes = sorted(set(CLASS_NAMES) - found_classes)
        if missing_classes:
            raise ValueError(
                f"Missing class folders in {root_dir}: {', '.join(missing_classes)}"
            )
        if not self.samples:
            raise ValueError(f"No PNG or JPG images found in {root_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        pixels = (pixels - mean) / std
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1))
        return tensor, torch.tensor(label, dtype=torch.long)


def split_training_data(
    dataset: MRIImageDataset, validation_fraction: float = 0.2, seed: int = 42
) -> Tuple[List[int], List[int]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    indices_by_class: Dict[int, List[int]] = {index: [] for index in range(len(CLASS_NAMES))}
    for index, (_, label) in enumerate(dataset.samples):
        indices_by_class[label].append(index)

    train_indices: List[int] = []
    validation_indices: List[int] = []
    generator = random.Random(seed)
    for label, indices in indices_by_class.items():
        if len(indices) < 2:
            raise ValueError(
                f"Need at least 2 training images per class; "
                f"{CLASS_NAMES[label]} has {len(indices)}."
            )
        generator.shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
    return train_indices, validation_indices
