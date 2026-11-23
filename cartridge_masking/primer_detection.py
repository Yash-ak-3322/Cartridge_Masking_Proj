"""Step 2 - locate the circular primer region.

The primer sits inside a bright circular gap (primer pocket) in the casing head, so a
Hough circle transform finds it reliably. If that fails (blurry scan, partial crop) we
fall back to a centred circle and report a warning instead of crashing.
"""
from typing import List, Tuple

import cv2
import numpy as np

from .config import MaskingConfig
from .models import Circle


def detect_primer_circle(gray: np.ndarray, cfg: MaskingConfig) -> Tuple[Circle, List[str]]:
    """Return the primer circle and a list of warnings."""
    h, w = gray.shape
    short = min(h, w)
    warnings: List[str] = []

    # Work on a downscaled copy: Hough is slow on multi-megapixel scans.
    scale = min(1.0, cfg.hough_work_size / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    small = cv2.medianBlur(small, 5)

    ss = min(small.shape)
    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=ss,  # minDist=ss -> a single best circle
        param1=100, param2=30,
        minRadius=int(cfg.min_radius_frac * ss), maxRadius=int(cfg.max_radius_frac * ss))

    if circles is not None:
        cx, cy, r = (circles[0, 0] / scale).tolist()
        off = np.hypot(cx - w / 2, cy - h / 2)
        if off <= cfg.max_center_offset_frac * short:
            return Circle(cx, cy, r, True), warnings
        warnings.append("Hough circle too far from image centre; using fallback primer circle.")
    else:
        warnings.append("No primer circle found by Hough; using fallback primer circle.")

    return Circle(w / 2, h / 2, cfg.fallback_radius_frac * short, False), warnings


def disc_mask(shape: Tuple[int, int], circle: Circle, radius_scale: float = 1.0) -> np.ndarray:
    """Boolean mask of a disc centred on the primer, optionally shrunk/enlarged."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - circle.cx) ** 2 + (yy - circle.cy) ** 2 <= (circle.r * radius_scale) ** 2
