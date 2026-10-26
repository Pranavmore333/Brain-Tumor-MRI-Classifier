import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Optional


class GradCAM:
    """Build a class activation map from gradients at a convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
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

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

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
        img_rgb = np.array(original_pil_image.convert("RGB"))
        h, w, _ = img_rgb.shape

        resized_cam = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * resized_cam)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        blended = np.uint8(alpha * heatmap_colored + (1.0 - alpha) * img_rgb)
        return Image.fromarray(blended)

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()
