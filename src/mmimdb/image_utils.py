"""Poster image restoration and simple image descriptors."""

from __future__ import annotations

import numpy as np
from PIL import Image

from mmimdb.constants import CAFFE_BGR_MEAN


def restore_poster_rgb(image_chw: np.ndarray, mean: tuple[float, float, float] = CAFFE_BGR_MEAN) -> np.ndarray:
    """Restore stored poster tensor to uint8 RGB image.

    The stored MM-IMDb poster tensor appears to be Caffe/VGG-style BGR pixels
    with channel means subtracted. This function reverses that representation
    for visualization, handcrafted descriptors, and XAI overlays.
    """
    image = image_chw.astype(np.float32).copy()
    image += np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    image = image[[2, 1, 0], :, :]
    image = np.transpose(image, (1, 2, 0))
    return np.clip(image, 0, 255).astype(np.uint8)


def rgb_to_model_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def color_histogram_descriptor(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    """Compute normalized global and 2x2 regional RGB histograms."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    regions = [(0, h, 0, w)]
    for y0, y1 in [(0, h // 2), (h // 2, h)]:
        for x0, x1 in [(0, w // 2), (w // 2, w)]:
            regions.append((y0, y1, x0, x1))

    parts = []
    for y0, y1, x0, x1 in regions:
        patch = rgb[y0:y1, x0:x1]
        for channel in range(3):
            hist, _ = np.histogram(patch[:, :, channel], bins=bins, range=(0, 255), density=False)
            hist = hist.astype(np.float32)
            denom = hist.sum()
            if denom > 0:
                hist /= denom
            parts.append(hist)
    return np.concatenate(parts).astype(np.float32)


def thumbnail_descriptor(rgb: np.ndarray, size: tuple[int, int] = (32, 20)) -> np.ndarray:
    """Compute a small grayscale thumbnail descriptor."""
    image = Image.fromarray(rgb, mode="RGB").convert("L").resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def image_descriptor(
    image_chw: np.ndarray,
    hist_bins: int = 16,
    thumbnail_size: tuple[int, int] = (32, 20),
) -> np.ndarray:
    rgb = restore_poster_rgb(image_chw)
    return np.concatenate(
        [
            color_histogram_descriptor(rgb, bins=hist_bins),
            thumbnail_descriptor(rgb, size=thumbnail_size),
        ]
    ).astype(np.float32)
