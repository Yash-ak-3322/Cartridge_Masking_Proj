# Automatic Masking of Cartridge-Case Head Scans

Detects and masks the primer region of a cartridge-case head scan and splits it into the
five regions required by the assignment, in the required order:

| # | Region | Colour | Output form |
|---|--------|--------|-------------|
| 1 | Breech-face impression | red | pixel mask |
| 2 | Aperture shear | green | pixel mask |
| 3 | Firing-pin shear | purple | pixel mask |
| 4 | Firing-pin impression | light blue | pixel mask |
| 5 | Firing-pin drag | blue arrow | vector (origin, direction, length) |

## Project layout
```
main.py                     CLI / PyCharm entry point
cartridge_masking/
  config.py                 all tunable parameters (relative to primer radius)
  models.py                 Label enum, Circle, DragVector, MaskResult
  io_utils.py               read/write images
  preprocessing.py          any input -> 8-bit grayscale
  primer_detection.py       Hough circle for the primer (+ fallback)
  firing_pin.py             smooth-region segmentation + drag direction
  mask_builder.py           builds the 5 regions in the required order
  pipeline.py               CartridgeMasker (single scan)
  multi_scan.py             MultiScanMasker (challenge: several scans of one case)
  visualization.py          overlays, legend, side-by-side images
  synthetic.py              synthetic scan generator (demo + tests)
tests/test_pipeline.py      pytest suite
outputs/                    sample side-by-side results
```

## Setup and run (PyCharm)
1. Open the folder as a project; create a Python 3.10+ interpreter (venv or conda).
2. `pip install -r requirements.txt` (or `conda env create -f environment.yml`).
3. Run `main.py` with no arguments for the demo, or use the CLI:
```
python main.py                                  # demo on synthetic scans -> outputs/
python main.py -i data/input -o outputs         # every image in a folder
python main.py -i scan.png                      # one scan
python main.py -i data/case_01 --multi-scan     # several scans of ONE cartridge case
pytest -q                                       # run tests
```
For each scan the program writes `<name>_comparison.png` (original | masked),
`<name>_labels.png` (label map: 0 bg, 1 breech, 2 aperture shear, 3 pin shear, 4 pin impression)
and `<name>_summary.json` (primer circle, drag angle, region areas, warnings).

## Methodology
1. **Preprocess** - convert gray/BGR/16-bit to 8-bit with a 1-99 percentile stretch, light blur.
2. **Primer circle** - Hough circle transform (on a downscaled copy) in a radius range expressed
   as a fraction of image size. Implausible/no result -> centred fallback circle + warning.
3. **Firing-pin impression** - the breech face is rough, the impression is smooth. A shading-removed
   local texture-energy map is thresholded (Otsu); the smooth blob that is large and close to the
   primer centre is kept, cleaned morphologically and hole-filled.
4. **Drag** - image moments give the blob's principal axis; the skewness along it points to the
   thin tail. The arrow starts at the blob centroid and points to the tail.
5. **Regions** - breech face = primer disc (edge trimmed) minus the dilated aperture; aperture
   shear = thin band hugging the back of the aperture; the impression is split at the drag
   origin into firing-pin shear (head half) and firing-pin impression (tail half).
   Later layers overwrite earlier ones, matching the required masking order.

All distances are relative to the primer radius R, so the method is resolution-independent and
adapts to other calibres (different primer sizes) without retuning. Different primer sizes and
rotations are covered by the demo/tests.

## Multi-scan challenge
`MultiScanMasker` masks each scan independently and fuses them by per-pixel majority vote, with
a median primer circle and circular-mean drag direction. A confidence map
(`fused_agreement_heatmap.png`) shows where scans disagree.

## Assumptions
- Input is a top-down head scan (intensity or shaded depth map), primer roughly centred, not cropped.
- Multi-scan inputs are already registered (same pixel grid).
- The head/tail split rule for firing-pin shear vs impression and the "back of aperture" rule for
  aperture shear are heuristics derived from the reference figure.

## Edge cases / limitations
- Featureless or very noisy input -> fallback circle/ellipse and a warning in the JSON/log.
- Primer with no visible gap ring may defeat Hough -> fallback circle.
- Strongly off-centre or partly cropped primers (> 25% offset) are rejected by the plausibility check.
- If the impression is not smooth (worn/rough pin) the texture cue weakens; tune `MaskingConfig`.
- **Validation status:** the bundled results and tests use SYNTHETIC scans (no real forensic data was
  available while developing). The heuristics, especially aperture shear, should be checked and tuned
  on real 9 mm scans (e.g. NIST Ballistics Toolmark Research Database) before relying on them.
