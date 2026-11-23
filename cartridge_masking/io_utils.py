"""Image input/output helpers."""
from pathlib import Path
from typing import List

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(path: Path) -> List[Path]:
    """Return a sorted list of image files for a file or a directory path."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Input path does not exist: {path}")


def read_image(path: Path) -> np.ndarray:
    """Read an image unchanged (keeps 16-bit depth maps intact)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    """Write an image, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise IOError(f"Could not write image: {path}")
