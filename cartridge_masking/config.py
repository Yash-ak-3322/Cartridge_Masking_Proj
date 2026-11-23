"""Central configuration.

Every geometric parameter is expressed RELATIVE to the primer radius R (or to the
image size) rather than in pixels. That is what lets the same code work for different
image resolutions (e.g. 1.75 um/px vs 3 um/px) and for different calibers whose primers
have different diameters.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MaskingConfig:
    # ---- Primer (outer circle) detection -------------------------------------------
    # Search range for the primer radius, as a fraction of min(image height, width).
    min_radius_frac: float = 0.25
    max_radius_frac: float = 0.50
    # Used when Hough detection fails: assume the primer is centred with this radius.
    fallback_radius_frac: float = 0.42
    # A detected circle whose centre is further than this fraction of the image size
    # from the image centre is considered implausible and rejected.
    max_center_offset_frac: float = 0.25
    # Hough runs on a downscaled copy (longest side) for speed.
    hough_work_size: int = 512
    # Fraction of R trimmed from the primer edge, to avoid the bright primer/casing gap.
    primer_edge_margin_frac: float = 0.07

    # ---- Smooth-region (firing pin aperture / impression) detection -----------------
    # Window of the local-texture-energy filter, as a fraction of min(image size).
    texture_window_frac: float = 0.015
    # Gaussian sigma (fraction of R) of the low-pass used to remove shading.
    shading_sigma_frac: float = 0.08
    # Only look for the impression within this fraction of R from the primer centre.
    search_radius_frac: float = 0.80
    # Candidate must cover at least / at most this fraction of the primer disc area.
    min_smooth_area_frac: float = 0.01
    max_smooth_area_frac: float = 0.60
    # Morphological closing kernel (fraction of R) used to merge fragments.
    close_kernel_frac: float = 0.06

    # ---- Derived regions -------------------------------------------------------------
    # Aperture-shear ring lies between these distances (fraction of R) from the impression.
    shear_inner_frac: float = 0.05
    shear_outer_frac: float = 0.14
    # Breech face excludes the impression dilated by this much (fraction of R).
    breech_exclusion_frac: float = 0.16
    # Shear band lateral half-width, relative to the impression half-width.
    shear_lateral_frac: float = 0.90
    # Drop breech-face fragments smaller than this fraction of the primer disc area.
    min_breech_component_frac: float = 0.01

    # ---- Visualisation ---------------------------------------------------------------
    overlay_alpha: float = 0.55
