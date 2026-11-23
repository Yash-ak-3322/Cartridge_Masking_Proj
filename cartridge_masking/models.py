"""Plain data containers shared by all modules."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple

import numpy as np


class Label(IntEnum):
    """Integer codes stored in the label map. The numeric order equals the order of
    masking required by the assignment (1 -> 4); later labels overwrite earlier ones.
    The firing-pin drag (5) is a vector, not a pixel region, so it lives in DragVector.
    """
    BACKGROUND = 0
    BREECH_FACE = 1          # red
    APERTURE_SHEAR = 2       # green
    FIRING_PIN_SHEAR = 3     # purple
    FIRING_PIN_IMPRESSION = 4  # light blue


@dataclass(frozen=True)
class Circle:
    """Primer outline: centre (cx, cy) and radius r, in pixels."""
    cx: float
    cy: float
    r: float
    detected: bool = True  # False when the fallback prior was used


@dataclass(frozen=True)
class DragVector:
    """Firing-pin drag: arrow starting at `origin`, pointing along unit vector `direction`."""
    origin: Tuple[float, float]
    direction: Tuple[float, float]
    length: float

    @property
    def angle_deg(self) -> float:
        """Direction angle in degrees, image coordinates (0 = right, 90 = down)."""
        return float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))


@dataclass
class MaskResult:
    """Everything produced for one scan."""
    name: str
    gray: np.ndarray                 # preprocessed 8-bit image, HxW
    label_map: np.ndarray            # uint8 HxW with values from `Label`
    circle: Circle
    drag: DragVector
    warnings: List[str] = field(default_factory=list)
    agreement: np.ndarray = None     # only set by MultiScanMasker (0..1 per pixel)

    def area(self, label: Label) -> int:
        """Pixel count of one label."""
        return int((self.label_map == int(label)).sum())

    def summary(self) -> dict:
        """JSON-serialisable summary for logging / reports."""
        return {
            "name": self.name,
            "primer_circle": {"cx": round(self.circle.cx, 1), "cy": round(self.circle.cy, 1),
                              "r": round(self.circle.r, 1), "detected": self.circle.detected},
            "drag": {"origin": [round(v, 1) for v in self.drag.origin],
                     "angle_deg": round(self.drag.angle_deg, 1),
                     "length_px": round(self.drag.length, 1)},
            "areas_px": {lab.name: self.area(lab) for lab in Label if lab != Label.BACKGROUND},
            "warnings": self.warnings,
        }
