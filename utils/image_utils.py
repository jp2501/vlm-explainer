# utils/image_utils.py

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def overlay_heatmap_on_image(image: Image.Image, cam: torch.Tensor, title: str = "") -> plt.Figure:
    """
    Returns a matplotlib Figure object with image + heatmap overlay.
    """
    patch_attn_np = cam.detach().cpu().numpy()
    H_img, W_img = image.size[1], image.size[0]
    patch_tensor = torch.from_numpy(patch_attn_np).unsqueeze(0).unsqueeze(0)

    upsampled = F.interpolate(
        patch_tensor,
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=False,
    )
    upsampled_np = upsampled[0, 0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.imshow(upsampled_np, alpha=0.5, cmap="jet")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def overlay_heatmap_to_array(image: Image.Image, cam: torch.Tensor) -> np.ndarray:
    """
    Returns an HxWx3 uint8 numpy array (RGB) of the image+heatmap overlay.
    This version does not rely on Matplotlib canvas, so it's safe for video/GIF creation.
    """
    # Base image as numpy
    img_np = np.array(image.convert("RGB"))  # [H, W, 3]
    H_img, W_img = img_np.shape[0], img_np.shape[1]

    # Upsample CAM to image size
    cam_np = cam.detach().cpu().numpy()
    cam_tensor = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0)  # [1,1,Hc,Wc]

    upsampled = F.interpolate(
        cam_tensor,
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=False,
    )
    heat = upsampled[0, 0].cpu().numpy()  # [H, W]

    # Normalize heatmap to [0, 1]
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    # Convert heatmap to RGB using a colormap (jet)
    colormap = cm.get_cmap("jet")
    heat_rgba = colormap(heat)  # [H, W, 4]
    heat_rgb = (heat_rgba[..., :3] * 255).astype(np.uint8)  # drop alpha, keep RGB

    # Blend heatmap with original image
    alpha = 0.5  # overlay strength
    frame = (alpha * heat_rgb + (1 - alpha) * img_np).astype(np.uint8)

    return frame
