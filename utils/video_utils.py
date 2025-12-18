# utils/video_utils.py

from typing import List
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import torch

from models.blip_explainer import BlipExplainer
from utils.image_utils import overlay_heatmap_to_array


def _ensure_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Ensure frame is HxWx3 uint8 RGB.
    """
    if frame is None:
        raise ValueError("Frame is None")

    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    # If RGBA, drop alpha
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Frame must be HxWx3, got shape {frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frame)


def _pad_to_even_hw(frame: np.ndarray) -> np.ndarray:
    """
    FFMPEG with yuv420p requires even width and height.
    Pad with black pixels (right/bottom) to next even numbers.
    """
    frame = _ensure_uint8_rgb(frame)
    h, w, _ = frame.shape

    new_h = h if (h % 2 == 0) else h + 1
    new_w = w if (w % 2 == 0) else w + 1

    if new_h == h and new_w == w:
        return frame

    pad_bottom = new_h - h
    pad_right = new_w - w

    padded = np.pad(
        frame,
        pad_width=((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return np.ascontiguousarray(padded)


def _add_layer_text_to_frame(frame: np.ndarray, layer_idx: int) -> np.ndarray:
    """
    Add 'Layer X' label at bottom-right.
    """
    frame = _ensure_uint8_rgb(frame)
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    text = f"Layer {layer_idx}"

    try:
        font = ImageFont.truetype("arial.ttf", size=36)
    except Exception:
        font = ImageFont.load_default()

    # Pillow compatibility: text size
    try:
        text_w, text_h = draw.textsize(text, font=font)  # older Pillow
    except Exception:
        bbox = draw.textbbox((0, 0), text, font=font)    # newer Pillow
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    W, H = pil_img.size
    padding = 18
    x = W - text_w - padding
    y = H - text_h - padding

    # black background box for readability
    bg_pad = 10
    draw.rectangle(
        [x - bg_pad, y - bg_pad, x + text_w + bg_pad, y + text_h + bg_pad],
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return np.array(pil_img)


def build_blip_layer_evolution_frames(
    blip: BlipExplainer,
    image: Image.Image,
    caption_ids: torch.Tensor,
    token_index: int,
) -> List[np.ndarray]:
    """
    Generate one base RGB frame per BLIP vision layer (Grad-CAM overlay).
    """
    base_frames: List[np.ndarray] = []
    for layer_idx in range(blip.num_vision_layers):
        cam = blip.gradcam_for_token(image, caption_ids, token_index, layer_idx=layer_idx)
        frame = overlay_heatmap_to_array(image, cam)
        frame = _ensure_uint8_rgb(frame)
        base_frames.append(frame)
    return base_frames


def _make_smooth_sequence(
    base_frames: List[np.ndarray],
    fps: int,
    seconds_per_layer: float,
    transition_seconds: float,
) -> List[np.ndarray]:
    """
    Build smooth sequence:
      - hold each layer for (seconds_per_layer - transition_seconds)
      - blend to next for transition_seconds
      - final layer holds for seconds_per_layer
    Label stays on the current layer during its hold+transition.
    """
    if not base_frames:
        return []

    frame_duration = 1.0 / fps
    interp_frames = max(0, int(round(transition_seconds / frame_duration)))

    hold_seconds_non_last = max(0.0, seconds_per_layer - transition_seconds)
    hold_frames_non_last = max(1, int(round(hold_seconds_non_last / frame_duration)))
    hold_frames_last = max(1, int(round(seconds_per_layer / frame_duration)))

    smooth: List[np.ndarray] = []
    n = len(base_frames)

    # Ensure consistent dtype/shape early
    base_frames = [_ensure_uint8_rgb(f) for f in base_frames]

    for i in range(n - 1):
        a = base_frames[i].astype(np.float32)
        b = base_frames[i + 1].astype(np.float32)

        # hold
        for _ in range(hold_frames_non_last):
            smooth.append(_add_layer_text_to_frame(base_frames[i], i))

        # transition
        for t in range(1, interp_frames + 1):
            alpha = t / (interp_frames + 1)
            blended = ((1 - alpha) * a + alpha * b).astype(np.uint8)
            smooth.append(_add_layer_text_to_frame(blended, i))

    # final layer hold
    last_idx = n - 1
    for _ in range(hold_frames_last):
        smooth.append(_add_layer_text_to_frame(base_frames[last_idx], last_idx))

    return smooth


def save_mp4(
    frames: List[np.ndarray],
    path: str,
    seconds_per_layer: float = 2.0,
    transition_seconds: float = 0.6,
    fps: int = 10,
) -> str:
    """
    Save MP4 safely:
      - pads frames to even width/height (required for yuv420p)
      - uses libx264 + yuv420p so Streamlit plays it everywhere
      - avoids macroblock resizing warnings without breaking ffmpeg
    """
    if not frames:
        raise ValueError("No frames provided for video.")

    # Build smooth timeline frames
    smooth_frames = _make_smooth_sequence(
        base_frames=frames,
        fps=fps,
        seconds_per_layer=seconds_per_layer,
        transition_seconds=transition_seconds,
    )

    # Pad every frame to even dimensions to prevent ffmpeg crash (Broken pipe)
    smooth_frames = [_pad_to_even_hw(f) for f in smooth_frames]

    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=1,  # stop imageio from forcing 16x blocks
    )

    try:
        for frame in smooth_frames:
            writer.append_data(frame)
    finally:
        writer.close()

    return path
