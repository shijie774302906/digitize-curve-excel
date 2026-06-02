# Digitize Curve Excel Skill

Codex skill for deterministic digitization of plotted curves from raster chart images into XLSX data, validation overlays, redraws, and accuracy audit reports.

The installable skill lives under:

```text
skills/digitize-curve-excel/
```

## What It Supports

- Fast batch extraction for exploratory multi-curve work.
- High-precision single-curve extraction with strict QA.
- Custom RGB/Lab color centers, data-coordinate ROI, and pixel ROI.
- Depth profiles, normal XY curves, dashed curves, straight references, marker series, and multi-series plots.
- Accuracy runner outputs with final XLSX, redraw, overlay, audit JSON, and next-step reports.

## Precision Modes

- `fast_batch`: faster, supports `multi_series`, but multi-series strict QA is limited.
- `high_precision`: slower, splits curves into single targets where possible and uses stricter mask/profile QA.

If a user explicitly accepts a redraw or overlay that still has conservative strict-QA failures, report it as `user_visual_accepted`; do not label it as strict PASS.

## Test Effect Gallery

Representative output examples are included under [`examples/test-effects/`](examples/test-effects/). These images show high-precision curve extraction, smoothed multi-series extraction, reference-line handling, and user-visual-accepted outputs.

### CPT Multi-Panel Overlay

![CPT multi-panel overlay result](examples/test-effects/cpt-multi-panel-overlay-result.png)

### CPT Before/After Panel Results

![CPT three-panel before/after](examples/test-effects/cpt-three-panel-before-after.png)

### Smoothed DSS Multi-Series Curves

![DSS smoothed multi-series result](examples/test-effects/dss-smoothed-multi-series-result.png)

### Reference Line Handling

![u0 reference line result](examples/test-effects/u0-reference-line-result.png)

### Single Depth-Profile Outputs

![Teal depth profile result](examples/test-effects/teal-depth-profile-result.png)

![Red depth profile result](examples/test-effects/red-depth-profile-result.png)

## Dependencies

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Validation

Run a basic syntax check:

```bash
python -m compileall skills/digitize-curve-excel/scripts
```

If you have benchmark fixtures, run:

```bash
python skills/digitize-curve-excel/scripts/run_regression_cases.py --benchmark-root path/to/workspace
```

The benchmark helper expects pre-existing `benchmark_runs/<case_id>/case_config.json` fixtures. Full private benchmark runs and generated audit directories are not included in this repository; the gallery above contains only a small set of representative test-effect images.

## Repository Notes

The skill folder intentionally contains only `SKILL.md`, `agents/`, and `scripts/`. Repository-level documentation, license, dependency files, and ignored generated artifacts live outside the skill folder.
