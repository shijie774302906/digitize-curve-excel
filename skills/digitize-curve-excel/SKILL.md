---
name: digitize-curve-excel
description: Digitize plotted curves from raster chart images into XLSX data, validation overlays, redraws, and accuracy-gated audit reports with explicit fast_batch versus high_precision mode selection. Use for single colored curves, multi-series plots, arbitrary-color high-precision single-curve extraction, depth-profile curves, dashed-line reconstruction, straight references, marker series, and benchmarked plot-to-data extraction from PNG/JPG figures.
---

# Digitize Curve Excel

Use deterministic image processing, not AI image generation. The output is a reproducible data table plus reviewable overlays/redraws. In `accuracy` mode, raw script output is only an attempt; the accuracy wrapper decides whether a target is PASS or FAIL.

## Required Precision Mode Confirmation

Before target confirmation or extraction, force the user to choose one user-facing precision mode. Do not infer this silently, even when the user says "extract all curves" or "run the benchmark".

- `fast_batch`: use for quick drafts, many curves, multi-series plots, and exploratory extraction. `multi_series` is allowed. Results may be visually useful and relatively accurate, but strict QA is limited for `multi_series`; final summaries must say `limited engine QA` when that warning appears.
- `high_precision`: use for final data, benchmark reproducibility, publication-quality digitization, or whenever the user asks for accuracy. Split curves into separate targets and prefer `single_curve` so strict mask/row-run/residual QA is available. Do not use `multi_series` as a fallback unless the user explicitly accepts `limited engine QA` for that target.

Ask this short question in the user's language before running a new extraction:

```text
请选择提取模式：
1. 快速批量：适合多曲线初稿，速度快，精度较高，但 multi-series 只有 limited QA。
2. 高精度：逐条曲线拆分为 single targets，速度慢，但用于最终可重复数据和更严格 QA。
```
Record the choice as top-level `precision_mode: "fast_batch"` or `precision_mode: "high_precision"` in the case config and repeat it in the user-facing summary. If the user chooses `high_precision` and a target cannot be handled by `single_curve`, stop and either add the needed single-curve color/ROI configuration or ask whether they accept `multi_series` limited QA for that specific target.

## Default Data Contract

Default every plotted curve to `data_form=continuous_curve` unless the user explicitly says the data should be treated as visible-only segments, a non-function-like visible path, discrete points, or a reference line.

Do not use `visible_segments`, `independent_axis=none`, or the derived `visible_polyline` target type as a fallback for difficult extraction, mask residuals, horizontal stroke extents, raster gaps, line thickness, marker contamination, or multi-series complexity. If a continuous target fails QA, repair the configuration or ask the user. Never silently downgrade a continuous curve to visible-path extraction.

Allowed visible-path cases:

- the user explicitly requests visible-only segments, broken segments, or no gap reconstruction
- a dashed target is explicitly confirmed as `dashed_handling=visible_only`
- the user explicitly says the target is not an `x -> y` or `y -> x` relationship and should be traced as a visible path
- strict QA proves the target is not single-valued under the declared axis and the user chooses visible-path digitization after being asked

When one of these exceptions is used, record an explicit confirmation field such as `visible_path_user_confirmed: true`, `visible_segments_user_confirmed: true`, or `non_function_path_user_confirmed: true`, plus a clear `user_confirmation_text`.

## Required Target Confirmation

Before running a new accuracy extraction, inspect the image and confirm only user-facing tags:

- precision mode: `fast_batch` or `high_precision`
- which visible curve(s) to digitize
- target color(s)
- data relationship: `x -> y`, `y -> x`, no function-like axis relationship, discrete points, or reference line
- line style: solid, dashed, dotted/marker, or mixed line+marker
- data form: continuous curve by default; visible-only segments only if explicitly requested
- whether same-color labels/text should be excluded

Do not ask the user to choose internal algorithm names such as `trace_mode`, `target_type`, or `single_value_axis`. Convert their answer into the user-intent fields below. `scripts/accuracy_runner.py` derives the internal contract from these fields before preflight and may overwrite stale internal values.

In `accuracy` mode, every target must record the user-intent fields:

- `confirmed_by_user: true`
- `confirmation_source`: `explicit_user_response` or `user_delegated_inference`
- `user_confirmation_text`: the user's answer or explicit delegation text
- `target_selection_note`
- `target_curves`: curve/series names and colors
- `independent_axis`: `x`, `y`, or `none`
- `dependent_axis`: `x`, `y`, or `none`; this may be derived from the declared relationship when omitted
- `line_style`: `solid`, `dashed`, `dotted`, `marker`, or `mixed_line_marker`
- `data_form`: `continuous_curve`, `visible_segments`, `discrete_points`, or `reference_line`
- `continuity_required`: `true` or `false`; this may be derived from `data_form`
- `target_colors`
- `exclude_same_color_text`

Do not set `confirmed_by_user: true` from agent visual inference alone. It is valid only when the user directly answers the target-confirmation questions, or when the user explicitly delegates the choices to the agent, for example "extract all main visible curves and infer colors/relationships." Record direct answers as `confirmation_source=explicit_user_response`; record explicit delegation as `confirmation_source=user_delegated_inference`. If the user only provides image paths or asks to run the benchmark, ask the confirmation questions first.

The runner derives the internal contract before preflight. These fields are not user questions and may be omitted from a fresh config:

- `curve_form`
- `single_value_axis`: `y_to_x`, `x_to_y`, `both`, or `none`
- `target_type` when it is omitted or conflicts with a mandatory form such as depth profile, marker series, or reference line
- depth-profile defaults such as `trace_mode=profile` and, for solid profiles, `profile_global_mask=true`

If any required user-intent field is missing or inconsistent, `scripts/accuracy_runner.py` fails preflight before extraction. If a stale internal field conflicts with the user-intent tags, the runner may overwrite it and record a contract warning in the audit.

Ask in the user's language, but keep the stored labels normalized. Use this prompt shape:

```text
1. Which curve(s) should be digitized, and what color is each?
2. Should I use fast_batch mode or high_precision mode?
3. What is the data relationship: x -> y, y -> x, no function-like path, discrete points, or reference line?
4. Is each target solid, dashed, dotted/marker, or mixed line+marker?
5. Unless you want visible-only segments or discrete/reference output, I will treat the data as continuous. Do you explicitly want any target kept as visible-only segments or a non-function visible path?
6. Should same-color labels/text be excluded from the data curve?
```

Tag mapping:

- `independent_axis=x`: x maps to y; the runner derives `dependent_axis=y`, `single_value_axis=x_to_y`, and each curve/series must have one representative y per x-bin.
- `independent_axis=y`: y maps to x; the runner derives `dependent_axis=x`, `single_value_axis=y_to_x`, and each curve/series must have one representative x per y/depth row.
- `independent_axis=none`: the target is not function-like; this is an exception path and requires explicit user confirmation. The runner derives `dependent_axis=none`, `single_value_axis=none`, and does not run x/y uniqueness checks.
- `data_form=continuous_curve`: the runner derives `continuity_required=true`; continuity, fragmentation, gaps, and mask residuals are hard checks.
- `data_form=visible_segments`: exception path only; the runner derives `continuity_required=false`; preserve visible gaps and do not invent connecting data.
- `data_form=discrete_points`: the runner derives `continuity_required=false`; audit marker/point extraction, not curve continuity.
- `data_form=reference_line`: use straight-reference auditing.
- `line_style=dashed`: ask whether to reconstruct a continuous curve or keep visible-only dashes before running.

If the audit shows that the selected curve is not single-valued under the declared independent axis, do not silently collapse or switch modes. First check for line thickness, labels, and marker artifacts. If the violation remains, ask the user whether to keep the declared axis and collapse to one representative value, switch the independent axis, or digitize the visible path without x/y function constraints.

Depth-axis sanity gate:

- If the y-axis label, target note, or visible axis text contains `Depth` or a local-language equivalent of depth/buried depth, treat the figure as a likely depth profile.
- Do not configure `independent_axis=x` for a likely depth profile unless the user explicitly confirms that x is the independent variable despite the depth y-axis. If that rare case is confirmed, record `depth_axis_user_confirmed=true` and explain it in `target_selection_note`.
- Otherwise record `independent_axis=y`; the runner derives `dependent_axis=x`, `single_value_axis=y_to_x`, and `curve_form=depth_profile_y_to_x`.
- A failed `x_to_y` run on a depth-looking figure is not a reason to collapse x-bins; it is a reason to revisit the axis confirmation.

Derived curve-form mapping:

- `depth_profile_y_to_x`: derived from `independent_axis=y` for continuous curves; requires `target_type=single_depth_profile`, `engine=single_curve`, and `trace_mode=profile`.
- `normal_xy_x_to_y`: derived from `independent_axis=x` for continuous curves. In high-precision single-curve mode this normally uses `target_type=single_xy_curve`; set `smooth_xy_curve` only when smoothness should be a hard audit.
- `smooth_path`: derived only when the user explicitly confirmed a non-function-like visible path.
- `visible_segments`: derived when the user explicitly wants visible-only segments; it preserves gaps instead of reconstructing continuity.

Supported `target_type` values:

- `single_depth_profile`
- `single_xy_curve`
- `dashed_depth_profile_continuous`
- `dashed_depth_profile_visible_only`
- `straight_reference`
- `straight_dashed_reference`
- `smooth_xy_curve`
- `multi_series_xy_curve`
- `multi_series_smooth_curve_only`
- `multi_series_depth_profile`
- `visible_polyline`
- `marker_series`
- `curve_with_markers`

## Choose A Mode

- `fast`: one extraction pass for rough drafts. You may run the low-level scripts directly, but do not present fast output as final accuracy output.
- `fast_batch` under `accuracy`: use `scripts/accuracy_runner.py` with `precision_mode=fast_batch`; `multi_series` is allowed, but any `strict_qa_limited_engine` warning must be surfaced to the user.
- `high_precision` under `accuracy`: use `scripts/accuracy_runner.py` with `precision_mode=high_precision`; split curves into separate `single_curve` targets where possible, use custom colors/ROI when needed, and do not publish a target as high precision if strict QA is limited.
- `accuracy`: fresh config, forced audit, optional react, and gated final output. Always run `scripts/accuracy_runner.py`; do not accept low-level script output directly.

Accuracy command:

```powershell
python path/to/digitize-curve-excel/scripts/accuracy_runner.py `
  --config "benchmark_runs/<case_id>/case_config.json"
```

Use `--strict-exit` when automation should return a nonzero exit code for any non-PASS case.

## Failure Handling Loop

Do not present `PASS` or `FAIL` as the user-facing outcome by itself. Treat status as an internal gate:

- If the run passes, give the final deliverables and mention the strict QA artifacts. If any target has `strict_qa_limited_engine`, explicitly state that it passed with limited engine QA and is not a high-precision strict-QA result.
- If the run fails with `remediation.next_action=repair_config_and_rerun`, apply the listed deterministic config fixes and rerun before talking to the user.
- If the run fails with `remediation.next_action=inspect_and_refine_config`, inspect `strict_qa.json`, `strict_qa_overlay.png`, masks, reports, and attempt artifacts; refine color masks, axes, ROI, trace mode, or tight exclusions, then rerun.
- If the run fails with `remediation.next_action=ask_user`, ask only the listed short user-facing questions. Do not ask about internal parameters.
- Stop only when final artifacts pass strict QA, or when the remaining blocker genuinely requires user target confirmation.

If the user explicitly reviews a redraw/overlay and accepts a target that still has strict QA failures, publish it only as `user_visual_accepted`, not as strict PASS. Record the accepted artifact path, the remaining failure codes, and the user's confirmation text in the final summary. This is appropriate for conservative QA failures caused by line thickness, markers, dashed fragments, same-color text, or a stricter-than-needed row-gap threshold; it is not appropriate when the extracted curve follows the wrong visible target.

Do not change a failed `continuous_curve` target to `visible_segments`, `independent_axis=none`, or `visible_polyline` during remediation unless the next step explicitly asks the user and the user confirms that semantic change.

The runner may automatically repair deterministic preflight mistakes once, such as `depth_profile_y_to_x` using `trend-profile` or a config relaxing hard depth-profile thresholds. These repaired attempts are written under `auto_repair_1/`.

After every accuracy run, read `benchmark_runs/<case_id>/next_steps.md` and follow it. That file converts audit details into the next workflow step:

- `ready_to_publish`: return the listed final artifacts to the user.
- `repair_config_and_rerun`: apply the listed fixes and rerun before responding.
- `inspect_and_refine_config`: inspect/refine artifacts first; ask the user only if the target cannot be safely classified.
- `ask_user`: ask the listed short questions directly and wait for the answer.

## Accuracy Config

Write a fresh `case_config.json` per case. Do not reuse previous configs or old trial scripts. The example below includes derived fields for readability, but a fresh config should treat `curve_form`, `single_value_axis`, `target_type`, `trace_mode`, and profile defaults as runner-derived unless the engine truly needs an explicit value.

```json
{
  "mode": "accuracy",
  "precision_mode": "high_precision",
  "case_id": "image_3",
  "input": "path/to/benchmark/image-3.png",
  "out_dir": "path/to/benchmark_runs/image_3",
  "max_react_passes": 2,
  "targets": [
    {
      "key": "dark_blue_solid",
      "name": "dark blue solid",
      "confirmed_by_user": true,
      "confirmation_source": "explicit_user_response",
      "user_confirmation_text": "User selected the dark blue solid curve only; relationship x -> y; solid continuous curve; exclude same-color labels/text.",
      "target_selection_note": "User selected the dark blue solid curve only.",
      "target_curves": [
        {"name": "dark blue solid", "color": "dark blue"}
      ],
      "independent_axis": "x",
      "dependent_axis": "y",
      "line_style": "solid",
      "data_form": "continuous_curve",
      "target_colors": ["dark blue"],
      "exclude_same_color_text": true,
      "target_type": "smooth_xy_curve",
      "engine": "single_curve",
      "script_args": {
        "axes": [0, 0, 0, 0],
        "x_min": 0,
        "x_max": 1,
        "y_min": 0,
        "y_max": 1,
        "curve_preset": "blue-solid",
        "trace_mode": "longest"
      },
      "forbidden_regions": [],
      "audit": {}
    }
  ]
}
```

For `single_curve`, `script_args` are converted to CLI flags for `scripts/digitize_curve_excel.py`. Use `exclude_rects` or `forbidden_regions` for labels, legends, same-color text, axes, and annotations.

For high-precision single-curve extraction with arbitrary colors, use custom single-curve color and ROI fields instead of switching to `multi_series`:

```json
"script_args": {
  "axes": [LEFT, RIGHT, TOP, BOTTOM],
  "x_min": 0,
  "x_max": 1,
  "y_min": 0,
  "y_max": 1,
  "trace_mode": "profile",
  "color_centers": [[182, 91, 175], [210, 155, 205]],
  "color_space": "lab",
  "max_color_dist": 65,
  "min_chroma": 12,
  "roi": [x_min, x_max, y_min, y_max]
}
```

Use `pixel_roi: [x0, y0, x1, y1]` when a data-coordinate ROI is inconvenient. The low-level single-curve script and strict QA both use the same custom color/ROI mask, so this path preserves strict audit consistency.

For high-precision normal `x -> y` curves, use `trace_mode: "x-profile"` with `guide_points` when markers, labels, or same-color nearby curves could pull the representative point away from the intended line:

```json
"script_args": {
  "trace_mode": "x-profile",
  "guide_points": [[0, 0], [5, 20], [10, 24]],
  "point_guide_tol_y": 3,
  "x_profile_interpolate_gap_px": 6
}
```

For depth profiles, do not use `trend-profile` when the contract is one x per depth y. Use `trace_mode=profile` and tight text/label exclusion boxes. Broad rectangles that cut through the target curve must fail strict QA.

For continuous solid depth profiles, prefer `profile_global_mask=true` with `trace_mode=profile` so the profile selector chooses one representative x from the whole target mask per depth row instead of restarting on each connected component. This is required when the target color is broken into many disconnected horizontal stroke components. Use a bounded `profile_interpolate_gap_rows` only when the source is confirmed continuous and the gap is caused by raster/color dropout, not when the user requested `visible_segments`.

When a `y -> x` profile collapses horizontal stroke extents to one x per depth row, strict QA may classify thin horizontal same-color leftovers as `collapsed_extra_run_components`. These are allowed only when the rows are already represented, the components are thin horizontal runs, and `extra_run_rows` stays within threshold. Larger same-color components, labels, or annotations must still fail or trigger user confirmation.

For `dashed_endpoint` and `multi_series`, provide `config`, `engine_config`, or a full engine config in `script_args`. The wrapper writes generated engine configs inside the attempt directory.

For multi-series depth profiles, do not default to `visible_polyline`. In `high_precision` mode, split each series into its own `single_depth_profile` target and use custom single-curve color/ROI if needed. If a shared `multi_series` engine is used for depth-profile curves, use the internal `target_type=multi_series_depth_profile`; every series must still preserve the declared `single_value_axis=y_to_x`; treat limited multi-series strict QA as weaker than `single_depth_profile` QA and state that limitation in the final summary. In `high_precision` mode, using `multi_series` requires explicit user acceptance of limited engine QA for that target.

For `multi_series` normal XY curves, use `target_type=multi_series_xy_curve` by default in `fast_batch` mode. Use `multi_series_smooth_curve_only` only when the target is expected to be smooth rather than stepped or kinked. Also set `script_args.single_value_axis="x_to_y"` or the same field per series. The multi-series engine collapses each series to one representative y per pixel x-bin, and the accuracy wrapper fails any remaining per-series x-to-multiple-y violation. Do not describe this as high-precision strict QA while `strict_qa_limited_engine` is present.

For normal XY continuous curves that include same-color markers or nearby scatter that should not be digitized, use `component_mode="x_profile"` with `guide_points` and `point_guide_tol_y` for each series. This samples one guide-nearest curve point per x-pixel column and prevents marker outlines from becoming vertical spikes in the final curve.

For low-contrast continuous curves where inspection shows a short missing pixel interval inside an otherwise visible curve, `x_profile` may set a small explicit `interpolate_gap_px` for that series. Do not use this for `visible_segments`, and do not use it to hide large missing curve sections; unresolved large gaps must remain blocked or become a user question.

For continuous curves, never publish a final result only because points exist. The final audit must show the curve passed the relevant continuity checks or, for engines where full mask QA is limited, the output must still satisfy the declared `single_value_axis` and low-level continuity/fragmentation diagnostics.

## Dash Rules

Dashed targets are not audited like solid curves.

- `dashed_depth_profile_continuous`: use `engine=dashed_endpoint`; every series must set `dash_mode=continuous`.
- `dashed_depth_profile_visible_only`: use `engine=dashed_endpoint`; every series must set `dash_mode=visible_only`.
- `straight_dashed_reference`: use `engine=dashed_endpoint`, `dash_mode=continuous`, and straight-line residual audit.

Continuous dashed profiles are checked for duplicate depth rows, row gaps, and mask residuals. Visible-only dashed profiles may have large row gaps because blank gaps are intentional.

## Low-Level Scripts

Use these scripts for fast mode, previews, or wrapper attempts:

- `scripts/digitize_curve_excel.py`: one colored curve in one calibrated frame.
- `scripts/digitize_dashed_endpoint_config.py`: dashed depth profiles by PCA endpoints.
- `scripts/digitize_multi_series_config.py`: multiple colored solid/dotted/dashed/marker series sharing one plot frame.

Single-curve example:

```powershell
python path/to/digitize-curve-excel/scripts/digitize_curve_excel.py `
  --input "figure.png" `
  --out-dir "outputs/curve_1" `
  --axes LEFT RIGHT TOP BOTTOM `
  --x-min 0 --x-max 8000 `
  --y-min 0 --y-max 16 `
  --reverse-y `
  --curve-preset red `
  --x-axis-position top `
  --trace-mode trend-profile
```

Use `--preview-masks` when target color is ambiguous. Supported presets are `red`, `green`, `blue`, `blue-solid`, `purple`, and `dark`. For arbitrary single-curve colors, use repeated `--color-center R G B` plus `--color-space rgb|lab`, `--max-color-dist`, `--min-chroma`, and `--roi` or `--pixel-roi`; preview output includes a `custom` mask when color centers are provided.

Use `--react-single-profile --react-max-passes 2` only for `profile` or `trend-profile` attempts. React changes selected row-level endpoints while preserving one `x` per `y` row.

## Accuracy Audit Rules

Accuracy mode has two gates:

1. low-level report parsing and numeric audit
2. strict mask/overlay QA before `final/` publication

Reactable failures:

- `duplicate_depth_rows > 0`
- `duplicate_y_rows > 0`
- continuous profile `max_row_gap > 2`
- profile `row_coverage < 0.90`
- configured residual ratio failure caused by missed curve residuals

React is capped at two passes. If the target still fails after two passes, stop and report FAIL.

Non-reactable failures:

- missing user confirmation fields
- unsafe profile threshold overrides
- low-level `CHECK` that still remains after allowed react attempts
- strict QA failures in `strict_qa.json`
- extracted points enter forbidden text/legend/axis regions
- `target_type` conflicts with the selected engine or dash mode
- a straight reference is extracted as a curve
- a smooth curve is jagged, broken, or backtracking beyond audit thresholds
- a non-marker target includes marker contamination or same-color text
- mask/color selection is visibly wrong or produces too few points

Non-reactable failures do not trigger react. Use `remediation` in the target audit to decide whether to repair config, inspect/refine artifacts, or ask the user. Do not stop at reporting the failure.

Default hard checks:

- forbidden region hits must be `0`
- depth/profile duplicate rows must be `0`
- continuous profile `max_row_gap <= 2`
- profile `row_coverage >= 0.90`
- low-level `Profile single-valuedness` status must be `PASS`
- accuracy configs must not relax `max_duplicate_depth_rows`, `max_profile_gap_rows`, or `min_profile_row_coverage`
- `single_value_axis=x_to_y` must have one representative y per x-bin within each series
- `continuity_required=true` must not pass with broken, highly fragmented, or visibly incomplete curve extraction
- straight reference line-fit RMS `<= 2.5 px` and p95 `<= 5 px`
- smooth curves check p95 second difference, spike fraction, and backtrack fraction; override thresholds in `audit` when needed

Strict QA semantics:

- `depth_profile_y_to_x`: compare each visible target-color mask row with the extracted row representative; fail missing rows, duplicate rows, representatives outside the row run, extra same-color runs, broad exclusions that remove long depth spans, and unclassified same-color components when `exclude_same_color_text=true`.
- `normal_xy_x_to_y`, `smooth_path`, and `visible_segments`: render the extracted path against the target-color mask and fail excessive residual mask coverage or large residual components.
- diagnostic overlay PNGs are review artifacts, not proof of PASS; `strict_qa.json` is the machine gate.

## Output Contract

For benchmark work, write only under `benchmark_runs/<case_id>/`.

```text
benchmark_runs/<case_id>/
  case_config.json
  attempts/
    <target_key>/
      attempt_0/
      auto_repair_1/
      react_1/
      react_2/
  final/
    <target_key>/
      result.xlsx
      overlay.png
      redrawn.png
      audit.json
      audit_summary.md
      strict_qa.json
      strict_qa_summary.md
      strict_qa_overlay.png
  audit.json
  audit_summary.md
  next_steps.md
```

`final/` is created only for PASS targets after strict QA passes. CSVs, masks, components, links, previews, and react diagnostics stay under `attempts/` by default. Failed targets must not leave stale or misleading final deliverables.

Root `next_steps.md` is the preferred workflow guide for the next user/agent action. Target `audit.json` also includes a `remediation` object with `next_action`, `automatic_fixes`, `inspection_steps`, and `user_questions`.

## Dependencies

Use a Python environment with:

```text
opencv-python
pillow
numpy
matplotlib
xlsxwriter
scikit-image
```
