"""Command-line entry point (also runs with NO arguments, ideal for PyCharm's Run button).

Examples
--------
python main.py                                   # demo on bundled synthetic scans
python main.py -i data/input -o outputs          # mask every image in a folder
python main.py -i scan.png                       # mask a single scan
python main.py -i data/case_01 --multi-scan      # fuse several scans of ONE cartridge case
"""
import argparse
import json
import logging
from pathlib import Path

import cv2

from cartridge_masking import CartridgeMasker, MaskingConfig, MultiScanMasker
from cartridge_masking.io_utils import list_images, read_image, write_image
from cartridge_masking.synthetic import make_cartridge_scan
from cartridge_masking.visualization import agreement_heatmap, side_by_side

log = logging.getLogger("cartridge_masking")
ROOT = Path(__file__).resolve().parent


def save_outputs(result, out_dir: Path, cfg: MaskingConfig) -> None:
    """Write side-by-side image, raw label map (PNG) and a JSON summary."""
    write_image(out_dir / f"{result.name}_comparison.png", side_by_side(result, cfg))
    write_image(out_dir / f"{result.name}_labels.png", result.label_map)  # values 0-4
    (out_dir / f"{result.name}_summary.json").write_text(json.dumps(result.summary(), indent=2))
    for w in result.warnings:
        log.warning("%s: %s", result.name, w)
    log.info("%s -> primer r=%.0fpx, drag angle=%.0f deg", result.name,
             result.circle.r, result.drag.angle_deg)


def make_demo_data(folder: Path) -> None:
    """Create synthetic scans of three 'calibres' (different primer sizes) + rotations."""
    specs = {"calibre_small_primer": (0.30, 25, 1), "calibre_medium_primer": (0.38, 70, 2),
             "calibre_large_primer": (0.44, 160, 3)}
    for name, (frac, ang, seed) in specs.items():
        write_image(folder / f"{name}.png", make_cartridge_scan(primer_frac=frac, angle_deg=ang, seed=seed))


def make_demo_multiscan(folder: Path) -> None:
    """Three registered 'scans' of the same case under different noise/brightness."""
    for i, (seed, b) in enumerate([(11, 0.0), (12, 0.05), (13, -0.05)], 1):
        write_image(folder / f"scan_{i}.png",
                    make_cartridge_scan(primer_frac=0.40, angle_deg=40, seed=seed, noise=1.2, brightness=b))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path, default=None, help="Image file or folder (default: demo).")
    p.add_argument("-o", "--output", type=Path, default=ROOT / "outputs", help="Output folder.")
    p.add_argument("--multi-scan", action="store_true",
                   help="Treat all images in --input as registered scans of ONE cartridge case.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = MaskingConfig()

    # Demo mode: nothing supplied -> generate synthetic data and run both modes.
    if args.input is None:
        log.info("No --input given: running demo on synthetic scans.")
        make_demo_data(ROOT / "data" / "input")
        make_demo_multiscan(ROOT / "data" / "multi_scan_case")
        masker = CartridgeMasker(cfg)
        for path in list_images(ROOT / "data" / "input"):
            save_outputs(masker.process_file(path), args.output / "single_scan", cfg)
        args.input, args.multi_scan, args.output = ROOT / "data" / "multi_scan_case", True, args.output / "multi_scan"

    paths = list_images(args.input)
    if not paths:
        raise SystemExit(f"No images found in {args.input}")

    if args.multi_scan:
        fused, per_scan = MultiScanMasker(cfg).process([read_image(q) for q in paths], [q.stem for q in paths])
        for r in per_scan:
            save_outputs(r, args.output, cfg)
        save_outputs(fused, args.output, cfg)
        write_image(args.output / "fused_agreement_heatmap.png", agreement_heatmap(fused))
    else:
        masker = CartridgeMasker(cfg)
        for path in paths:
            save_outputs(masker.process_file(path), args.output, cfg)
    log.info("Done. Results in %s", args.output)


if __name__ == "__main__":
    main()
