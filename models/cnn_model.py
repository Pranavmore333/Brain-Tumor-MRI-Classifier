import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


class CustomBrainTumorCNN(nn.Module):
    """
    Custom 4-Layer Convolutional Neural Network designed for Brain Tumor MRI classification.

    Architecture:
    - 4 Conv Blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool2D)
    - Feature extraction from 3 -> 32 -> 64 -> 128 -> 256 channels
    - Adaptive Average Pooling to fixed (4x4) spatial dimension
    - Fully Connected Classifier with Dropout regularization (p=0.4, p=0.3)
    """
    def __init__(self, num_classes: int = 4):
        super(CustomBrainTumorCNN, self).__init__()

        # Block 1: Low-level edge and texture detection
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: Structural contour and shape detection
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: Anatomical region & ventricular boundary detection
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4: High-level tumor lesion & tissue anomaly detection
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Conv Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Global pooling and classification
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def get_custom_cnn(num_classes: int = 4) -> CustomBrainTumorCNN:
    """Helper factory function to instantiate the custom CNN."""
    return CustomBrainTumorCNN(num_classes=num_classes)
