"""Challenge: mask ONE cartridge case from MULTIPLE scans.

ASSUMPTION: the scans are already registered (same pixel grid), e.g. several lighting
conditions / a depth map and an intensity map of the same head. Each scan is masked
independently, then results are fused by per-pixel MAJORITY VOTE. The fraction of scans that
agree is stored as a confidence map (`MaskResult.agreement`). Scans of different size are
resized (nearest-neighbour for labels) to the first scan's grid.
"""
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import MaskingConfig
from .models import Circle, DragVector, Label, MaskResult
from .pipeline import CartridgeMasker


class MultiScanMasker:
    def __init__(self, config: Optional[MaskingConfig] = None):
        self.masker = CartridgeMasker(config)

    def process(self, images: Sequence[np.ndarray], names: Sequence[str]
                ) -> Tuple[MaskResult, List[MaskResult]]:
        """Return (fused_result, per_scan_results)."""
        if len(images) == 0:
            raise ValueError("At least one scan is required.")
        per_scan = [self.masker.process(im, n) for im, n in zip(images, names)]
        ref_shape = per_scan[0].label_map.shape

        # Bring every label map onto the reference grid.
        stack = []
        for r in per_scan:
            lm = r.label_map
            if lm.shape != ref_shape:
                lm = cv2.resize(lm, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_NEAREST)
            stack.append(lm)
        stack = np.stack(stack)                                   # (N, H, W)

        # Majority vote per pixel.
        counts = np.stack([(stack == int(l)).sum(0) for l in Label])  # (L, H, W)
        fused = counts.argmax(0).astype(np.uint8)
        agreement = counts.max(0) / float(len(per_scan))

        # Fuse geometry: median circle, circular-mean drag direction.
        circle = Circle(float(np.median([r.circle.cx for r in per_scan])),
                        float(np.median([r.circle.cy for r in per_scan])),
                        float(np.median([r.circle.r for r in per_scan])),
                        all(r.circle.detected for r in per_scan))
        vec = np.sum([r.drag.direction for r in per_scan], axis=0)
        norm = np.linalg.norm(vec)
        warnings = [f"[{r.name}] {w}" for r in per_scan for w in r.warnings]
        if norm < 0.5 * len(per_scan):
            warnings.append("Drag directions disagree across scans; fused arrow is unreliable.")
        vec = vec / norm if norm > 1e-9 else np.array([1.0, 0.0])
        drag = DragVector(
            origin=(float(np.median([r.drag.origin[0] for r in per_scan])),
                    float(np.median([r.drag.origin[1] for r in per_scan]))),
            direction=(float(vec[0]), float(vec[1])),
            length=float(np.median([r.drag.length for r in per_scan])))

        fused_result = MaskResult(name="fused", gray=per_scan[0].gray, label_map=fused,
                                  circle=circle, drag=drag, warnings=warnings, agreement=agreement)
        return fused_result, per_scan
