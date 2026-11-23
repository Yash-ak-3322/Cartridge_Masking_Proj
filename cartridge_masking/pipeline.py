"""Single-scan pipeline: preprocess -> primer circle -> impression -> drag -> label map."""
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import MaskingConfig
from .firing_pin import estimate_drag, segment_impression
from .io_utils import read_image
from .mask_builder import build_label_map
from .models import MaskResult
from .preprocessing import preprocess
from .primer_detection import detect_primer_circle


class CartridgeMasker:
    """Masks one cartridge-case head scan.

    Example
    -------
    >>> masker = CartridgeMasker()
    >>> result = masker.process_file("scan.png")
    >>> result.label_map  # uint8 array, values from models.Label
    """

    def __init__(self, config: Optional[MaskingConfig] = None):
        self.cfg = config or MaskingConfig()

    def process(self, image: np.ndarray, name: str = "scan") -> MaskResult:
        gray = preprocess(image)                                         # 1. normalise
        circle, warn1 = detect_primer_circle(gray, self.cfg)             # 2. primer outline
        impression, warn2 = segment_impression(gray, circle, self.cfg)   # 3. smooth blob
        drag = estimate_drag(impression)                                 #    drag axis
        labels = build_label_map(gray.shape, circle, impression, drag, self.cfg)  # 4. regions
        return MaskResult(name=name, gray=gray, label_map=labels,
                          circle=circle, drag=drag, warnings=warn1 + warn2)

    def process_file(self, path: Union[str, Path]) -> MaskResult:
        path = Path(path)
        return self.process(read_image(path), name=path.stem)
