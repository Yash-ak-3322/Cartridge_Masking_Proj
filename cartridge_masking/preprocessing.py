"""Step 1 - turn any input scan into a clean 8-bit grayscale image."""
import cv2
import numpy as np


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Convert gray / BGR / BGRA / 16-bit input to 8-bit grayscale.

    Uses a 1-99 percentile stretch, so a few extreme pixels (specular glints, or the
    white void around a rotated scan, as in the reference image) do not crush contrast.
    """
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99))
    if hi - lo < 1e-6:  # flat image: avoid division by zero
        return np.zeros(img.shape, np.uint8)
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Normalise + light denoise. Returns an 8-bit grayscale image."""
    gray = to_gray_u8(img)
    return cv2.GaussianBlur(gray, (3, 3), 0)
