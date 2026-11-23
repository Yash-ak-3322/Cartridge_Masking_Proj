"""Step 3 - find the firing-pin impression and its drag direction.

Key observation: the breech face is MACHINED/ROUGH (high local texture) while the
firing-pin impression is a SMOOTH depression. So we compute a local texture-energy map
and keep the smooth blob nearest the primer centre. Image moments then give the blob's
principal axis; the skewness along that axis tells us which end is the elongated
"drag" tail.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import MaskingConfig
from .models import Circle, DragVector
from .primer_detection import disc_mask


def _odd(n: int, minimum: int = 3) -> int:
    n = max(int(n), minimum)
    return n if n % 2 == 1 else n + 1


def texture_energy(gray: np.ndarray, circle: Circle, cfg: MaskingConfig) -> np.ndarray:
    """Local high-frequency energy. Low values = smooth surface."""
    g = gray.astype(np.float32) / 255.0
    sigma = max(circle.r * cfg.shading_sigma_frac, 3.0)
    high_pass = g - cv2.GaussianBlur(g, (0, 0), sigma)       # remove slow shading/curvature
    win = _odd(cfg.texture_window_frac * min(gray.shape), 5)
    return cv2.blur(high_pass * high_pass, (win, win))        # local mean of squared detail


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes by redrawing external contours filled."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def segment_impression(gray: np.ndarray, circle: Circle,
                       cfg: MaskingConfig) -> Tuple[np.ndarray, List[str]]:
    """Return a boolean mask of the smooth firing-pin impression region."""
    warnings: List[str] = []
    h, w = gray.shape
    search = disc_mask(gray.shape, circle, cfg.search_radius_frac)
    disc_area = float(disc_mask(gray.shape, circle).sum())

    energy = texture_energy(gray, circle, cfg)
    # Log-scale + 8-bit so Otsu's threshold behaves well on the skewed distribution.
    e = np.log1p(energy * 1e4)
    e8 = cv2.normalize(e, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vals = e8[search].reshape(-1, 1)
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smooth = (e8 <= thr) & search

    # Clean up: remove speckle, merge fragments.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(circle.r * 0.03),) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(circle.r * cfg.close_kernel_frac),) * 2)
    m = cv2.morphologyEx(smooth.astype(np.uint8), cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

    # Choose the blob that is big AND close to the primer centre.
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    best: Optional[int] = None
    best_score = -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < cfg.min_smooth_area_frac * disc_area:
            continue
        dist = np.hypot(cents[i][0] - circle.cx, cents[i][1] - circle.cy)
        score = area / (1.0 + dist / circle.r)
        if score > best_score:
            best, best_score = i, score

    if best is not None:
        mask = _fill_holes(lab == best)
        frac = mask.sum() / disc_area
        if cfg.min_smooth_area_frac <= frac <= cfg.max_smooth_area_frac:
            return mask, warnings
        warnings.append(f"Impression covers {frac:.0%} of primer (implausible); using fallback.")
    else:
        warnings.append("No smooth impression blob found; using fallback ellipse.")

    # Fallback prior: a small ellipse at the primer centre.
    fb = np.zeros((h, w), np.uint8)
    cv2.ellipse(fb, (int(circle.cx), int(circle.cy)),
                (int(circle.r * 0.30), int(circle.r * 0.20)), 0, 0, 360, 1, -1)
    return fb.astype(bool), warnings


def estimate_drag(mask: np.ndarray) -> DragVector:
    """Principal axis + skewness -> drag arrow.

    * centroid / orientation come from second-order image moments;
    * the sign of the 3rd moment (skewness) of pixel positions along the axis points
      toward the long, thin tail, which is the direction the pin dragged.
    """
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    mu20 = ((xs - cx) ** 2).mean()
    mu02 = ((ys - cy) ** 2).mean()
    mu11 = ((xs - cx) * (ys - cy)).mean()
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    d = np.array([np.cos(theta), np.sin(theta)])

    t = (xs - cx) * d[0] + (ys - cy) * d[1]
    if (t ** 3).mean() < 0:      # make the arrow point toward the tail
        d = -d
        t = -t
    return DragVector(origin=(float(cx), float(cy)),
                      direction=(float(d[0]), float(d[1])),
                      length=float(t.max()))
