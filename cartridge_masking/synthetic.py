"""Synthetic cartridge-head generator.

Real forensic scans are not redistributable, so this module draws plausible fake ones for
the unit tests and for the bundled demo: rough breech-face texture, a bright primer/casing
gap, a smooth teardrop-shaped firing-pin impression with a known drag direction, and a
striated band (aperture shear) at its back. Primer size, rotation and noise are parameters,
which also lets us demonstrate generalisation over different calibres.
"""
import cv2
import numpy as np


def make_cartridge_scan(size: int = 512, primer_frac: float = 0.40, angle_deg: float = 20.0,
                        seed: int = 0, noise: float = 1.0, brightness: float = 0.0) -> np.ndarray:
    """Return an 8-bit grayscale synthetic scan.

    primer_frac : primer radius as a fraction of image size (varies with calibre)
    angle_deg   : true drag direction (0 = right, 90 = down)
    """
    rng = np.random.default_rng(seed)
    c = size / 2.0
    R = primer_frac * size
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    rr = np.hypot(xx - c, yy - c)

    def tex(sigma, amp):
        return cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), sigma) * amp

    img = np.full((size, size), 0.62, np.float32) + tex(1.5, 0.35 * noise)            # casing
    primer = rr <= R
    img[primer] = 0.50 + tex(1.0, 0.55 * noise)[primer]                              # rough breech face
    ring = (rr > R) & (rr < R * 1.07)
    img[ring] = 0.97                                                                  # bright gap

    th = np.radians(angle_deg)
    d = np.array([np.cos(th), np.sin(th)])
    a = 0.20 * R                                                                      # head radius
    head_c = np.array([c, c]) - d * 0.25 * R
    imp = np.zeros((size, size), np.uint8)
    cv2.circle(imp, tuple(int(v) for v in head_c), int(a), 1, -1)
    tail_c = head_c + d * 1.0 * a
    cv2.ellipse(imp, tuple(int(v) for v in tail_c), (int(1.9 * a), int(0.62 * a)), angle_deg, 0, 360, 1, -1)
    imp = imp.astype(bool)

    # Smooth dark depression with soft shading.
    shade = cv2.GaussianBlur(imp.astype(np.float32), (0, 0), 0.12 * R)
    img[imp] = (0.28 + 0.25 * shade[imp]) + tex(2.0, 0.01)[imp]
    # Striated band (aperture shear) behind the impression.
    s = (xx - head_c[0]) * -d[1] + (yy - head_c[1]) * d[0]
    t = (xx - head_c[0]) * d[0] + (yy - head_c[1]) * d[1]
    band = (t < -1.25 * a) & (t > -1.75 * a) & (np.abs(s) < 1.0 * a) & primer
    img[band] = 0.55 + 0.25 * np.sin(s[band] * 0.9)

    img = np.clip(img + brightness, 0, 1)
    return (img * 255).astype(np.uint8)
