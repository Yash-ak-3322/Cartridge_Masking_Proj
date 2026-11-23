"""Automatic masking of cartridge-case head scans.

Public API
----------
CartridgeMasker: mask a single scan.
MultiScanMasker: mask several registered scans of ONE cartridge case and fuse them.
MaskingConfig: all tunable parameters in one place.
Label: integer codes used in the label map.
"""
from .config import MaskingConfig
from .models import Circle, DragVector, Label, MaskResult
from .pipeline import CartridgeMasker
from .multi_scan import MultiScanMasker

__all__ = ["CartridgeMasker", "MultiScanMasker", "MaskingConfig",
           "Circle", "DragVector", "Label", "MaskResult"]
