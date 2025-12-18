# utils/patch_utils.py

from typing import Set, Tuple
import numpy as np
from PIL import Image


def rect_patch_indices(row_start: int, row_end: int,
                       col_start: int, col_end: int) -> Set[Tuple[int, int]]:
    """
    Returns all (row, col) pairs in the inclusive rectangle
    [row_start, row_end] x [col_start, col_end].
    """
    indices: Set[Tuple[int, int]] = set()
    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):
            indices.add((r, c))
    return indices


def mask_patches_from_indices(
    image: Image.Image,
    patch_indices: Set[Tuple[int, int]],
    Hp: int,
    Wp: int,
    mask_color: tuple[int, int, int] = (150, 150, 150),
) -> Image.Image:
    """
    Mask patches specified by (row, col) indices.
    Hp, Wp: patch grid size (same as cam.shape).
    """
    img_np = np.array(image).copy()  # [H, W, 3]
    H_img, W_img = img_np.shape[0], img_np.shape[1]

    patch_h = H_img // Hp
    patch_w = W_img // Wp

    for (row, col) in patch_indices:
        y0 = row * patch_h
        y1 = (row + 1) * patch_h
        x0 = col * patch_w
        x1 = (col + 1) * patch_w

        img_np[y0:y1, x0:x1] = mask_color

    return Image.fromarray(img_np)
