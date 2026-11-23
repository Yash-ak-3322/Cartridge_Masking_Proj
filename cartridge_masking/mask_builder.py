"""Step 4 - build the labelled regions in the order required by the assignment.

Order of masking (later layers overwrite earlier ones):
    1 breech-face impression (red)
    2 aperture shear         (green)
    3 firing-pin shear       (purple)
    4 firing-pin impression  (light blue)
    5 firing-pin drag        (blue arrow, handled as a vector in DragVector)

"""
import cv2
import numpy as np

from .config import MaskingConfig
from .models import Circle, DragVector, Label
from .primer_detection import disc_mask


def _dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(max(radius_px, 1)) + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def build_label_map(shape, circle: Circle, impression: np.ndarray,
                    drag: DragVector, cfg: MaskingConfig) -> np.ndarray:
    """Assemble the uint8 label map from the primer circle, impression and drag axis."""
    R = circle.r
    labels = np.zeros(shape, np.uint8)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    dx, dy = drag.direction
    ox, oy = drag.origin
    t = (xx - ox) * dx + (yy - oy) * dy          # position along the drag axis
    s = -(xx - ox) * dy + (yy - oy) * dx         # position across the drag axis

    # ---- 1. Breech face: primer disc minus the aperture neighbourhood ---------------
    primer = disc_mask(shape, circle, 1.0 - cfg.primer_edge_margin_frac)
    breech = primer & ~_dilate(impression, cfg.breech_exclusion_frac * R)
    # Drop tiny fragments so the red mask is clean.
    n, comp, stats, _ = cv2.connectedComponentsWithStats(breech.astype(np.uint8), connectivity=8)
    min_area = cfg.min_breech_component_frac * float(primer.sum())
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            breech[comp == i] = False
    labels[breech] = Label.BREECH_FACE

    # ---- 2. Aperture shear: ring around the impression, restricted to its back side ---
    ring = _dilate(impression, cfg.shear_outer_frac * R) & ~_dilate(impression, cfg.shear_inner_frac * R)
    t_imp, s_imp = t[impression], s[impression]
    back_extent = max(-t_imp.min(), 1.0)
    half_width = max(np.abs(s_imp).max(), 1.0)
    behind = t < -0.5 * back_extent
    narrow = np.abs(s) < cfg.shear_lateral_frac * half_width
    labels[ring & behind & narrow & primer] = Label.APERTURE_SHEAR

    # ---- 3. / 4. Split impression at the drag origin --------------------------------
    labels[impression & (t <= 0)] = Label.FIRING_PIN_SHEAR
    labels[impression & (t > 0)] = Label.FIRING_PIN_IMPRESSION
    return labels
