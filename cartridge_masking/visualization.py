"""Rendering: coloured overlay, legend and side-by-side comparison images."""
from typing import Tuple

import cv2
import numpy as np

from .config import MaskingConfig
from .models import Label, MaskResult

# BGR colours following the assignment figure.
COLORS = {
    Label.BREECH_FACE: (0, 0, 255),               # red
    Label.APERTURE_SHEAR: (0, 200, 0),            # green
    Label.FIRING_PIN_SHEAR: (200, 0, 160),        # purple
    Label.FIRING_PIN_IMPRESSION: (255, 230, 80),  # light blue
}
ARROW_COLOR = (255, 0, 0)                         # blue
LEGEND = [("1 Breech face", Label.BREECH_FACE), ("2 Aperture shear", Label.APERTURE_SHEAR),
          ("3 Firing pin shear", Label.FIRING_PIN_SHEAR),
          ("4 Firing pin impression", Label.FIRING_PIN_IMPRESSION)]


def render_overlay(result: MaskResult, cfg: MaskingConfig = MaskingConfig()) -> np.ndarray:
    """Original image tinted with the masks and the drag arrow (BGR)."""
    base = cv2.cvtColor(result.gray, cv2.COLOR_GRAY2BGR)
    colour = base.copy()
    for lab, bgr in COLORS.items():
        colour[result.label_map == int(lab)] = bgr
    out = cv2.addWeighted(colour, cfg.overlay_alpha, base, 1 - cfg.overlay_alpha, 0)
    # Keep the unmasked area untouched for clarity.
    out[result.label_map == 0] = base[result.label_map == 0]

    # Primer outline (thin white) + drag arrow with a small circle at the origin.
    thick = max(2, result.gray.shape[0] // 250)
    cv2.circle(out, (int(result.circle.cx), int(result.circle.cy)), int(result.circle.r),
               (255, 255, 255), 1, cv2.LINE_AA)
    ox, oy = result.drag.origin
    dx, dy = result.drag.direction
    tip = (int(ox + dx * result.drag.length), int(oy + dy * result.drag.length))
    cv2.arrowedLine(out, (int(ox), int(oy)), tip, ARROW_COLOR, thick, cv2.LINE_AA, tipLength=0.15)
    cv2.circle(out, (int(ox), int(oy)), thick * 3, ARROW_COLOR, thick, cv2.LINE_AA)
    return out


def _legend_strip(width: int) -> np.ndarray:
    """White bar with colour swatches describing each mask."""
    strip = np.full((36, width, 3), 255, np.uint8)
    x = 10
    for text, lab in LEGEND + [("5 Drag", None)]:
        bgr = COLORS.get(lab, ARROW_COLOR)
        cv2.rectangle(strip, (x, 11), (x + 14, 25), bgr, -1)
        cv2.putText(strip, text, (x + 20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        x += 20 + 9 * len(text) + 18
    return strip


def side_by_side(result: MaskResult, cfg: MaskingConfig = MaskingConfig()) -> np.ndarray:
    """[ original | masked overlay ] with titles and a legend."""
    left = cv2.cvtColor(result.gray, cv2.COLOR_GRAY2BGR)
    right = render_overlay(result, cfg)
    gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
    row = np.hstack([left, gap, right])
    title = np.full((30, row.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"{result.name}  -  original", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(title, "masked", (left.shape[1] + 18, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    legend = _legend_strip(row.shape[1])
    return np.vstack([title, row, legend])


def agreement_heatmap(result: MaskResult) -> np.ndarray:
    """Multi-scan confidence map (fraction of scans agreeing per pixel) as a colour image."""
    return cv2.applyColorMap((result.agreement * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
