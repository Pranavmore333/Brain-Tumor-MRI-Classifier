import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) Generator.

    Produces visual heatmaps explaining which anatomical regions in the brain MRI
    contributed most strongly to the neural network's classification decision.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates normalized Grad-CAM heatmap array for a specific class index.

        Args:
            input_tensor: Tensor of shape (1, C, H, W)
            class_idx: Target class index. If None, uses model's predicted class.

        Returns:
            heatmap: 2D numpy array of shape (H, W) with values in range [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        self.gradients = None
        self.activations = None

        with torch.enable_grad():
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            output[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture model activations.")

        # Global average pooling of gradients across spatial dimensions (H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)

        # Weighted linear combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H, W)
        cam = F.relu(cam) # Apply ReLU to focus on features with positive influence

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_on_image(
        self,
        original_pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Image.Image:
        """
        Overlays the computed Grad-CAM heatmap on top of the original MRI image.

        Args:
            original_pil_image: PIL Image of the original MRI scan.
            heatmap: 2D numpy array [0, 1].
            alpha: Transparency weight for blending (0 = pure image, 1 = pure heatmap).
            colormap: OpenCV colormap constant (default: COLORMAP_JET).

        Returns:
            Blended PIL Image showing the highlighted tumor region.
        """
        # Convert original image to RGB numpy array
        img_rgb = np.array(original_pil_image.convert("RGB"))
        h, w, _ = img_rgb.shape

        # Resize heatmap to match image dimensions
        resized_cam = cv2.resize(heatmap, (w, h))

        # Convert to 8-bit unsigned integer (0-255)
        heatmap_uint8 = np.uint8(255 * resized_cam)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Alpha blend original image and heatmap
        blended = np.uint8(alpha * heatmap_colored + (1.0 - alpha) * img_rgb)
        return Image.fromarray(blended)

    def remove_hooks(self):
        """Clean up hook handles."""
        self.forward_handle.remove()
        self.backward_handle.remove()
