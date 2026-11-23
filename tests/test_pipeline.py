"""Unit tests on synthetic scans with known ground truth. Run:  pytest -q"""
import numpy as np
import pytest

from cartridge_masking import CartridgeMasker, Label, MultiScanMasker
from cartridge_masking.synthetic import make_cartridge_scan


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


@pytest.mark.parametrize("frac,angle", [(0.30, 25), (0.38, 70), (0.44, 160), (0.40, -120)])
def test_single_scan_geometry(frac, angle):
    """Primer radius and drag direction are recovered across primer sizes/rotations."""
    res = CartridgeMasker().process(make_cartridge_scan(primer_frac=frac, angle_deg=angle, seed=1))
    assert res.circle.detected
    assert abs(res.circle.r - frac * 512) < 0.12 * frac * 512
    assert _angle_diff(res.drag.angle_deg, angle) < 15


def test_all_labels_present_and_ordered():
    res = CartridgeMasker().process(make_cartridge_scan(seed=2))
    for lab in Label:
        if lab != Label.BACKGROUND:
            assert res.area(lab) > 0, f"{lab.name} missing"
    assert res.label_map.dtype == np.uint8 and res.label_map.max() <= 4


def test_blank_image_does_not_crash():
    """Edge case: featureless input must fall back gracefully with warnings."""
    res = CartridgeMasker().process(np.full((256, 256), 128, np.uint8))
    assert res.warnings


def test_color_and_16bit_inputs():
    g = make_cartridge_scan(seed=3)
    CartridgeMasker().process(np.dstack([g, g, g]))
    CartridgeMasker().process(g.astype(np.uint16) * 256)


def test_multi_scan_fusion():
    imgs = [make_cartridge_scan(angle_deg=40, seed=s, noise=1.2) for s in (11, 12, 13)]
    fused, per = MultiScanMasker().process(imgs, ["a", "b", "c"])
    assert len(per) == 3 and fused.agreement.shape == fused.label_map.shape
    assert _angle_diff(fused.drag.angle_deg, 40) < 15
