from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency diagnostic.
    raise SystemExit(f"Missing dependency: numpy. Original error: {exc}")

from strict_qa import run_strict_qa


SCRIPT_DIR = Path(__file__).resolve().parent

SUPPORTED_TARGET_TYPES = {
    "single_depth_profile",
    "single_xy_curve",
    "dashed_depth_profile_continuous",
    "dashed_depth_profile_visible_only",
    "straight_reference",
    "straight_dashed_reference",
    "smooth_xy_curve",
    "multi_series_xy_curve",
    "multi_series_smooth_curve_only",
    "multi_series_depth_profile",
    "visible_polyline",
    "marker_series",
    "curve_with_markers",
}

DASHED_TARGET_TYPES = {
    "dashed_depth_profile_continuous",
    "dashed_depth_profile_visible_only",
    "straight_dashed_reference",
}
STRAIGHT_TARGET_TYPES = {"straight_reference", "straight_dashed_reference"}
SMOOTH_TARGET_TYPES = {"smooth_xy_curve", "multi_series_smooth_curve_only", "curve_with_markers"}
PROFILE_TARGET_TYPES = {"single_depth_profile"}
MULTI_SERIES_PROFILE_TARGET_TYPES = {"multi_series_depth_profile"}

REACTABLE_CODES = {
    "profile_report_check",
    "profile_duplicate_depth_rows",
    "profile_max_row_gap",
    "profile_row_coverage",
    "profile_residual_ratio",
}

CURVE_FORMS = {
    "depth_profile_y_to_x",
    "normal_xy_x_to_y",
    "smooth_path",
    "visible_segments",
}

SINGLE_VALUE_AXES = {"x_to_y", "y_to_x", "both", "none"}
INDEPENDENT_AXES = {"x", "y", "none"}
DEPENDENT_AXES = {"x", "y", "none"}
CONFIRMATION_SOURCES = {"explicit_user_response", "user_delegated_inference"}
LINE_STYLES = {"solid", "dashed", "dotted", "marker", "mixed_line_marker"}
DATA_FORMS = {"continuous_curve", "visible_segments", "discrete_points", "reference_line"}
DEPTH_AXIS_RE = re.compile(r"(depth|\u6df1\u5ea6|\u57cb\u6df1)", re.IGNORECASE)

CONFIRMATION_FIELDS = {
    "confirmed_by_user",
    "confirmation_source",
    "user_confirmation_text",
    "target_selection_note",
    "target_curves",
    "independent_axis",
    "dependent_axis",
    "line_style",
    "dashed_handling",
    "data_form",
    "visible_path_user_confirmed",
    "visible_segments_user_confirmed",
    "non_function_path_user_confirmed",
    "same_color_text_user_confirmed",
    "text_occlusion_user_confirmed",
    "target_colors",
    "exclude_same_color_text",
}

AUTO_REPAIR_PREFLIGHT_CODES = {
    "depth_profile_requires_profile_mode",
    "unsafe_threshold_override",
}

USER_CONFIRMATION_CODES = {
    "missing_confirmation",
    "missing_confirmation_source",
    "invalid_confirmation_source",
    "missing_user_confirmation_text",
    "missing_target_selection_note",
    "missing_target_curves",
    "missing_independent_axis",
    "missing_dependent_axis",
    "missing_line_style",
    "missing_data_form",
    "missing_target_colors",
    "missing_exclude_same_color_text",
    "axis_mapping_mismatch",
    "data_form_continuity_mismatch",
    "dashed_mode_needs_confirmation",
    "visible_path_confirmation_required",
    "curve_form_axis_mismatch",
    "depth_axis_confirmation_required",
    "contract_derivation_failed",
}

MISSING_USER_CONFIRMATION_CODES = {
    "missing_confirmation",
    "missing_confirmation_source",
    "invalid_confirmation_source",
    "missing_user_confirmation_text",
    "missing_target_selection_note",
    "missing_target_curves",
    "missing_independent_axis",
    "missing_dependent_axis",
    "missing_line_style",
    "missing_data_form",
    "missing_target_colors",
    "missing_exclude_same_color_text",
    "contract_derivation_failed",
}

PROFILE_RE = re.compile(
    r"Profile single-valuedness: status=(?P<status>\w+); "
    r"selected_depth_rows=(?P<selected_rows>\d+); "
    r"curve_mask_rows=(?P<mask_rows>\d+); "
    r"row_coverage=(?P<row_coverage>[-+0-9.eE]+); "
    r"duplicate_depth_rows=(?P<duplicate_depth_rows>\d+); "
    r"max_row_gap=(?P<max_row_gap>[-+0-9.eE]+); "
    r"p95_row_dx=(?P<p95_row_dx>[-+0-9.eE]+) px; "
    r"p95_second_diff=(?P<p95_second_diff>[-+0-9.eE]+) px\."
)
CONTINUITY_RE = re.compile(
    r"Continuity: status=(?P<status>\w+); "
    r"segments=(?P<segments>\d+); "
    r"largest_segment_ratio=(?P<largest_segment_ratio>[-+0-9.eE]+); "
    r"short_segments=(?P<short_segments>\d+); "
    r"internal_gaps=(?P<internal_gaps>\d+); "
    r"max_internal_gap=(?P<max_internal_gap>[-+0-9.eE]+)"
)
REACT_RE = re.compile(
    r"Single-profile react pass (?P<pass_index>\d+)/(?P<max_passes>\d+): "
    r"accepted_candidates=(?P<accepted_candidates>\d+); "
    r"corrected_rows=(?P<corrected_rows>\d+); "
    r"residual_ratio=(?P<base_residual_ratio>[-+0-9.eE]+)->(?P<reacted_residual_ratio>[-+0-9.eE]+); "
    r"duplicate_y_count=(?P<duplicate_y_count>\d+)\."
)


def slugify(value: str, default: str = "target") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or default


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def threshold(audit: dict[str, Any], key: str, default: float) -> float:
    return to_float(audit.get(key, default), default)


def bool_opt(target: dict[str, Any], audit: dict[str, Any], key: str, default: bool = False) -> bool:
    if key in audit:
        return bool(audit[key])
    if key in target:
        return bool(target[key])
    return default


def infer_engine(target: dict[str, Any], target_type: str) -> str:
    raw = str(target.get("engine", "")).strip().lower()
    aliases = {
        "": "",
        "single": "single_curve",
        "curve": "single_curve",
        "single_curve": "single_curve",
        "dashed": "dashed_endpoint",
        "dashed_endpoint": "dashed_endpoint",
        "endpoint": "dashed_endpoint",
        "multi": "multi_series",
        "multi_series": "multi_series",
    }
    engine = aliases.get(raw)
    if engine:
        return engine
    if target_type in DASHED_TARGET_TYPES:
        return "dashed_endpoint"
    if target_type in {"multi_series_xy_curve", "multi_series_smooth_curve_only", "multi_series_depth_profile"}:
        return "multi_series"
    return "single_curve"


def add_cli_arg(cmd: list[str], key: str, value: Any) -> None:
    flag = key if key.startswith("--") else "--" + key.replace("_", "-")
    if value is None or value is False:
        return
    if value is True:
        cmd.append(flag)
        return
    if key in {"guide_point", "guide_points", "guide"}:
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            points = value
        else:
            points = [value]
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            cmd.append("--guide-point")
            cmd.extend(str(part) for part in point)
        return
    if key in {"color_center", "color_centers", "centers"}:
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            centers = value
        else:
            centers = [value]
        for center in centers:
            if not isinstance(center, (list, tuple)) or len(center) != 3:
                continue
            cmd.append("--color-center")
            cmd.extend(str(part) for part in center)
        return
    if key in {"exclude_rect", "exclude_rects"}:
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            rects = value
        else:
            rects = [value]
        for rect in rects:
            cmd.append("--exclude-rect")
            cmd.extend(str(part) for part in rect)
        return
    if isinstance(value, (list, tuple)):
        cmd.append(flag)
        cmd.extend(str(part) for part in value)
        return
    cmd.extend([flag, str(value)])


def build_single_curve_command(
    input_path: Path,
    attempt_dir: Path,
    target: dict[str, Any],
    react_passes: int | None,
) -> list[str]:
    script_args = dict(target.get("script_args") or {})
    if str(script_args.get("trace_mode", "")).lower() == "skeleton":
        script_args["trace_mode"] = "longest"
    stem = slugify(str(script_args.pop("stem", target.get("key", "curve"))), "curve")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "digitize_curve_excel.py"),
        "--input",
        str(input_path),
        "--out-dir",
        str(attempt_dir),
        "--stem",
        stem,
    ]
    if react_passes is not None:
        cmd.extend(["--react-single-profile", "--react-max-passes", str(react_passes)])
    skip = {"input", "out_dir", "out-dir", "stem", "react_single_profile", "react_max_passes"}
    for key, value in script_args.items():
        if key in skip:
            continue
        add_cli_arg(cmd, key, value)
    return cmd


def materialize_engine_config(target: dict[str, Any], attempt_dir: Path, config_base: Path) -> Path:
    candidates = [
        target.get("engine_config"),
        target.get("config_json"),
        target.get("config"),
        (target.get("script_args") or {}).get("config") if isinstance(target.get("script_args"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            config_path = attempt_dir / f"{slugify(str(target.get('key', 'target')))}_engine_config.json"
            write_json(config_path, candidate)
            return config_path
        if isinstance(candidate, str):
            return resolve_path(candidate, config_base)

    script_args = target.get("script_args")
    if isinstance(script_args, dict) and any(key in script_args for key in ("series", "output_prefix", "axes")):
        config_path = attempt_dir / f"{slugify(str(target.get('key', 'target')))}_engine_config.json"
        write_json(config_path, script_args)
        return config_path

    raise ValueError("dashed_endpoint and multi_series targets require config, engine_config, or script_args containing the engine config")


def validate_dash_mode(config_path: Path, target_type: str) -> list[dict[str, Any]]:
    if target_type not in DASHED_TARGET_TYPES:
        return []
    expected = "visible_only" if target_type == "dashed_depth_profile_visible_only" else "continuous"
    config = read_json(config_path)
    failures: list[dict[str, Any]] = []
    for index, series in enumerate(config.get("series", []), start=1):
        mode = str(series.get("dash_mode", "")).strip().lower()
        if mode != expected:
            failures.append(
                failure(
                    "dash_mode_mismatch",
                    f"series {series.get('key', index)!r} must set dash_mode={expected!r} for target_type={target_type}",
                    reactable=False,
                )
            )
    if not config.get("series"):
        failures.append(failure("dash_config_no_series", "dashed endpoint config has no series", reactable=False))
    return failures


def build_config_engine_command(
    engine: str,
    input_path: Path,
    attempt_dir: Path,
    config_path: Path,
) -> list[str]:
    script_name = "digitize_dashed_endpoint_config.py" if engine == "dashed_endpoint" else "digitize_multi_series_config.py"
    return [
        sys.executable,
        str(SCRIPT_DIR / script_name),
        "--input",
        str(input_path),
        "--config",
        str(config_path),
        "--out-dir",
        str(attempt_dir),
    ]


def run_command(cmd: list[str], attempt_dir: Path) -> int:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    write_json(attempt_dir / "command.json", {"command": cmd})
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR.parent), text=True, capture_output=True, env=env)
    (attempt_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (attempt_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    return int(proc.returncode)


def failure(code: str, message: str, reactable: bool) -> dict[str, Any]:
    return {"code": code, "message": message, "reactable": bool(reactable)}


def warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def unique_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def failure_codes(failures: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("code", "")) for item in failures}


def build_remediation(
    failures: list[dict[str, Any]],
    metrics: dict[str, Any],
    reactable: bool,
) -> dict[str, Any]:
    codes = failure_codes(failures)
    automatic_fixes: list[str] = []
    user_questions: list[str] = []
    inspection_steps: list[str] = []

    if not failures:
        return {
            "next_action": "ready_to_publish",
            "summary": "Strict QA passed; publish final artifacts.",
            "automatic_fixes": [],
            "user_questions": [],
            "inspection_steps": [],
        }

    if codes & MISSING_USER_CONFIRMATION_CODES:
        user_questions.extend(
            [
                "Which curve(s) should be digitized, and what color is each?",
                "What is the data relationship: x -> y, y -> x, no function-like path, discrete points, or reference line?",
                "Is each target solid, dashed, dotted/marker, or mixed line+marker?",
                "Is the data a continuous curve, visible-only segments, discrete points, or a reference line?",
                "Should same-color labels/text be excluded from the data curve?",
            ]
        )
    if "depth_axis_confirmation_required" in codes:
        user_questions.insert(0, "The y-axis appears to be depth. Is depth/y the independent variable for this target, or did you explicitly intend x to be independent?")
    if "dashed_mode_needs_confirmation" in codes:
        user_questions.append("For dashed targets, should the dashes be reconstructed as a continuous curve or kept as visible-only segments?")
    if "visible_path_confirmation_required" in codes:
        user_questions.append(
            "This target would be treated as visible-only segments or a non-function visible path. Did you explicitly want that, or should it remain a continuous x-y/y-x curve?"
        )

    if "depth_profile_requires_profile_mode" in codes:
        automatic_fixes.append("Set script_args.trace_mode to profile for depth_profile_y_to_x and rerun.")
    if "unsafe_threshold_override" in codes:
        automatic_fixes.append("Remove relaxed depth-profile audit thresholds and rerun with hard defaults.")
    if "profile_report_check" in codes and reactable:
        automatic_fixes.append("Run the capped profile react loop, then audit the react attempt.")
    if "profile_report_check" in codes and not reactable:
        inspection_steps.append("Inspect the low-level calibration report and strict_qa metrics before changing extraction parameters.")

    if codes & {"strict_depth_missing_rows", "strict_depth_missing_span", "strict_depth_row_run_coverage", "profile_max_row_gap", "profile_row_coverage"}:
        inspection_steps.extend(
            [
                "Open strict_qa_overlay.png and compare the target-color mask against the extracted line.",
                "Check whether exclude_rects cut through the data curve; shrink or split broad exclusions, then rerun.",
                "If the missing interval is caused by a label or annotation, add a tight same-color text exclusion instead of a broad rectangle.",
            ]
        )
    if codes & {"strict_unclassified_same_color_component", "strict_depth_extra_runs", "strict_depth_duplicate_rows", "strict_depth_representative_outside_run"}:
        inspection_steps.append("Use strict_qa.json component samples to add tight exclude_rects for same-color text or refine the color preset.")
        user_questions.append("For any same-color component flagged in strict_qa.json, is it data curve or text/annotation?")
    if codes & {"too_few_points", "missing_profile_metrics", "strict_mask_residual_ratio", "strict_mask_residual_component"}:
        inspection_steps.extend(
            [
                "Run mask previews for candidate color presets and choose the preset that isolates only the target curve.",
                "Verify axes/crop bounds and ROI before rerunning.",
            ]
        )
    if "single_value_axis_violation" in codes:
        inspection_steps.append(
            "Verify that line thickness, labels, and marker artifacts are not causing the single-value violation."
        )
        user_questions.append(
            "The selected curve is not single-valued under the declared independent axis. Should it keep that axis and collapse to one representative value, switch the independent axis, or be digitized as a visible path without x/y function constraints?"
        )
    if codes & {"continuity_axis_gap", "continuity_report_check"}:
        inspection_steps.append("For continuous_curve targets, refine the mask/ROI or line extraction until each series is continuous along the independent axis.")
    if "forbidden_region_hit" in codes:
        inspection_steps.append("Tighten target selection or exclude regions so extracted points do not enter known text/legend/axis regions.")
    if "script_exit_nonzero" in codes:
        inspection_steps.append("Read stderr.txt and command.json from the attempt directory, fix the script/config error, then rerun.")

    if user_questions:
        next_action = "ask_user"
        summary = "Need a short user confirmation before a reliable result can be produced."
    elif automatic_fixes:
        next_action = "repair_config_and_rerun"
        summary = "Apply the deterministic config repair(s), then rerun accuracy mode."
    else:
        next_action = "inspect_and_refine_config"
        summary = "Use strict QA diagnostics to refine exclusions, color mask, axes, or target definition, then rerun."

    return {
        "next_action": next_action,
        "summary": summary,
        "automatic_fixes": unique_items(automatic_fixes),
        "user_questions": unique_items(user_questions),
        "inspection_steps": unique_items(inspection_steps),
        "strict_qa_summary": ((metrics.get("strict_qa") or {}).get("metrics") or {}),
    }


def overall_next_action(targets: list[dict[str, Any]]) -> str:
    actions = [(target.get("remediation") or {}).get("next_action", "") for target in targets]
    if actions and all(action == "ready_to_publish" for action in actions):
        return "ready_to_publish"
    if "repair_config_and_rerun" in actions:
        return "repair_config_and_rerun"
    if "inspect_and_refine_config" in actions:
        return "inspect_and_refine_config"
    if "ask_user" in actions:
        return "ask_user"
    return "inspect_and_refine_config"


def collect_unique_remediation_items(targets: list[dict[str, Any]], key: str) -> list[str]:
    items: list[str] = []
    for target in targets:
        remediation = target.get("remediation") or {}
        for item in remediation.get(key) or []:
            items.append(str(item))
    return unique_items(items)


def refresh_attempt_remediation(audit: dict[str, Any], reactable: bool | None = None) -> None:
    effective_reactable = bool(audit.get("reactable")) if reactable is None else bool(reactable)
    audit["reactable"] = effective_reactable
    audit["remediation"] = build_remediation(list(audit.get("failures") or []), dict(audit.get("metrics") or {}), effective_reactable)
    attempt_dir = Path(str(audit.get("attempt_dir", "")))
    if attempt_dir:
        write_json(attempt_dir / "audit.json", audit)
        write_attempt_summary(attempt_dir / "audit_summary.md", audit)


def parse_reports(attempt_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    react_passes: list[dict[str, Any]] = []
    for report_path in sorted(attempt_dir.rglob("*report*.txt")) + sorted(attempt_dir.rglob("*calibration_report.txt")):
        text = report_path.read_text(encoding="utf-8", errors="replace")
        profile_match = PROFILE_RE.search(text)
        if profile_match:
            values = profile_match.groupdict()
            metrics["profile"] = {
                "status": values["status"],
                "selected_rows": to_int(values["selected_rows"]),
                "mask_rows": to_int(values["mask_rows"]),
                "row_coverage": to_float(values["row_coverage"]),
                "duplicate_depth_rows": to_int(values["duplicate_depth_rows"]),
                "max_row_gap": to_float(values["max_row_gap"]),
                "p95_row_dx": to_float(values["p95_row_dx"]),
                "p95_second_diff": to_float(values["p95_second_diff"]),
                "report": str(report_path),
            }
        continuity_match = CONTINUITY_RE.search(text)
        if continuity_match:
            values = continuity_match.groupdict()
            metrics["continuity"] = {
                "status": values["status"],
                "segments": to_int(values["segments"]),
                "largest_segment_ratio": to_float(values["largest_segment_ratio"]),
                "short_segments": to_int(values["short_segments"]),
                "internal_gaps": to_int(values["internal_gaps"]),
                "max_internal_gap": to_float(values["max_internal_gap"]),
                "report": str(report_path),
            }
        for match in REACT_RE.finditer(text):
            values = match.groupdict()
            react_passes.append(
                {
                    "pass_index": to_int(values["pass_index"]),
                    "max_passes": to_int(values["max_passes"]),
                    "accepted_candidates": to_int(values["accepted_candidates"]),
                    "corrected_rows": to_int(values["corrected_rows"]),
                    "base_residual_ratio": to_float(values["base_residual_ratio"]),
                    "reacted_residual_ratio": to_float(values["reacted_residual_ratio"]),
                    "duplicate_y_count": to_int(values["duplicate_y_count"]),
                    "report": str(report_path),
                }
            )
    if react_passes:
        metrics["react_passes"] = react_passes
        metrics["last_reacted_residual_ratio"] = react_passes[-1]["reacted_residual_ratio"]
    return metrics


def read_csv_points(attempt_dir: Path) -> np.ndarray:
    rows: list[tuple[float, float, int]] = []
    seen: set[tuple[str, int, float, float]] = set()
    for csv_path in sorted(attempt_dir.rglob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "pixel_x" not in reader.fieldnames or "pixel_y" not in reader.fieldnames:
                    continue
                for index, row in enumerate(reader):
                    px = to_float(row.get("pixel_x"))
                    py = to_float(row.get("pixel_y"))
                    if not math.isfinite(px) or not math.isfinite(py):
                        continue
                    order = to_int(row.get("point_order", index), index)
                    key = (str(csv_path), order, round(px, 3), round(py, 3))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append((px, py, order))
        except OSError:
            continue
    rows.sort(key=lambda item: item[2])
    if not rows:
        return np.empty((0, 2), dtype=float)
    return np.array([[px, py] for px, py, _ in rows], dtype=float)


def single_value_axis_metrics(attempt_dir: Path, axis: str, tolerance_px: float = 1.5) -> dict[str, Any]:
    if axis not in {"x_to_y", "y_to_x"}:
        return {"axis": axis, "checked": False}
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for csv_path in sorted(attempt_dir.rglob("*.csv")):
        if not csv_path.name.endswith("_digitized.csv"):
            continue
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "pixel_x" not in reader.fieldnames or "pixel_y" not in reader.fieldnames:
                    continue
                for row in reader:
                    px = to_float(row.get("pixel_x"))
                    py = to_float(row.get("pixel_y"))
                    if not math.isfinite(px) or not math.isfinite(py):
                        continue
                    series = str(row.get("series_key") or row.get("segment_id") or csv_path.stem)
                    bucket = int(round(px if axis == "x_to_y" else py))
                    grouped.setdefault((series, bucket), []).append((px, py))
        except OSError:
            continue

    duplicate_bins = 0
    max_spread = 0.0
    samples: list[dict[str, Any]] = []
    for (series, bucket), points_for_bucket in grouped.items():
        if len(points_for_bucket) <= 1:
            continue
        values = [py for _px, py in points_for_bucket] if axis == "x_to_y" else [px for px, _py in points_for_bucket]
        spread = float(max(values) - min(values))
        if spread <= tolerance_px:
            continue
        duplicate_bins += 1
        max_spread = max(max_spread, spread)
        if len(samples) < 10:
            samples.append({"series_key": series, "pixel_bin": bucket, "point_count": len(points_for_bucket), "spread_px": spread})

    return {
        "axis": axis,
        "checked": True,
        "pixel_bin_count": int(len(grouped)),
        "duplicate_bins": int(duplicate_bins),
        "max_spread_px": float(max_spread),
        "tolerance_px": float(tolerance_px),
        "samples": samples,
    }


def axis_continuity_metrics(attempt_dir: Path, axis: str, max_gap_px: float = 20.0) -> dict[str, Any]:
    if axis not in {"x_to_y", "y_to_x"}:
        return {"axis": axis, "checked": False}
    series_points: dict[str, list[tuple[float, float]]] = {}
    for csv_path in sorted(attempt_dir.rglob("*.csv")):
        if not csv_path.name.endswith("_digitized.csv"):
            continue
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "pixel_x" not in reader.fieldnames or "pixel_y" not in reader.fieldnames:
                    continue
                for row in reader:
                    px = to_float(row.get("pixel_x"))
                    py = to_float(row.get("pixel_y"))
                    if not math.isfinite(px) or not math.isfinite(py):
                        continue
                    series = str(row.get("series_key") or row.get("segment_id") or csv_path.stem)
                    series_points.setdefault(series, []).append((px, py))
        except OSError:
            continue

    max_gap = 0.0
    gap_count = 0
    samples: list[dict[str, Any]] = []
    for series, points_for_series in series_points.items():
        if len(points_for_series) < 2:
            continue
        values = sorted((px if axis == "x_to_y" else py) for px, py in points_for_series)
        gaps = [float(b - a) for a, b in zip(values, values[1:])]
        if not gaps:
            continue
        local_max = max(gaps)
        max_gap = max(max_gap, local_max)
        for gap in gaps:
            if gap > max_gap_px:
                gap_count += 1
                if len(samples) < 10:
                    samples.append({"series_key": series, "gap_px": float(gap)})

    return {
        "axis": axis,
        "checked": True,
        "series_count": int(len(series_points)),
        "max_gap_px": float(max_gap),
        "gap_count": int(gap_count),
        "max_allowed_gap_px": float(max_gap_px),
        "samples": samples,
    }


def forbidden_hits(points: np.ndarray, regions: list[Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if points.size == 0:
        return hits
    px = points[:, 0]
    py = points[:, 1]
    for index, region in enumerate(regions, start=1):
        if isinstance(region, dict):
            name = str(region.get("name", f"region_{index}"))
            rect = region.get("rect")
        else:
            name = f"region_{index}"
            rect = region
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            continue
        x0, y0, x1, y1 = [to_float(part) for part in rect]
        left, right = sorted([x0, x1])
        top, bottom = sorted([y0, y1])
        count = int(np.count_nonzero((px >= left) & (px <= right) & (py >= top) & (py <= bottom)))
        if count:
            hits.append({"name": name, "rect": [left, top, right, bottom], "hit_count": count})
    return hits


def line_metrics(points: np.ndarray) -> dict[str, float | int]:
    if len(points) < 2:
        return {"point_count": int(len(points)), "rms_px": float("inf"), "p95_px": float("inf"), "max_px": float("inf")}
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]])
    distances = np.abs(centered @ normal)
    return {
        "point_count": int(len(points)),
        "rms_px": float(np.sqrt(np.mean(distances**2))),
        "p95_px": float(np.percentile(distances, 95)),
        "max_px": float(np.max(distances)),
    }


def smooth_metrics(points: np.ndarray) -> dict[str, float | int]:
    if len(points) < 5:
        return {
            "point_count": int(len(points)),
            "p95_second_diff_px": 0.0,
            "spike_fraction": 0.0,
            "backtrack_fraction": 0.0,
        }
    diff = np.diff(points, axis=0)
    step = np.linalg.norm(diff, axis=1)
    second = np.diff(points, n=2, axis=0)
    second_norm = np.linalg.norm(second, axis=1)
    median_step = float(np.median(step)) if len(step) else 0.0
    spike_floor = max(8.0, 4.0 * median_step)
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[0]
    delta_projection = np.diff(projection)
    forward = 1.0 if np.sum(delta_projection) >= 0 else -1.0
    backtracks = np.count_nonzero(forward * delta_projection < -1.0)
    return {
        "point_count": int(len(points)),
        "p95_second_diff_px": float(np.percentile(second_norm, 95)),
        "spike_fraction": float(np.count_nonzero(second_norm > spike_floor) / max(len(second_norm), 1)),
        "backtrack_fraction": float(backtracks / max(len(delta_projection), 1)),
    }


def parse_dashed_summaries(attempt_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(attempt_dir.rglob("*_endpoint_summary.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                summaries.append(
                    {
                        "series_key": row.get("series_key", ""),
                        "dash_mode": row.get("dash_mode", ""),
                        "duplicate_y_rows": to_int(row.get("duplicate_y_rows")),
                        "mask_residual_p95_px": to_float(row.get("mask_residual_p95_px")),
                        "max_y_gap_rows": to_float(row.get("max_y_gap_rows")),
                        "profile_rows": to_int(row.get("profile_rows")),
                        "summary_csv": str(path),
                    }
                )
    return summaries


def find_artifact(attempt_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(
            (path for path in attempt_dir.rglob(pattern) if path.is_file()),
            key=lambda path: (len(path.parts), str(path).lower()),
        )
        if matches:
            return matches[0]
    return None


def collect_artifacts(attempt_dir: Path, engine: str) -> dict[str, str | None]:
    xlsx_patterns = (
        ["*_digitized_redrawn.xlsx", "*_endpoint_profiles.xlsx", "*_endpoint_profile.xlsx", "*.xlsx"]
        if engine != "multi_series"
        else ["*.xlsx"]
    )
    overlay_patterns = (
        ["*_redraw_overlay.png", "*_endpoint_overlay.png", "*_overlay.png"]
        if engine != "multi_series"
        else ["*_overlay.png", "*_mask_overlap.png"]
    )
    redrawn_patterns = (
        ["*_redrawn.png", "*_endpoint_redraw.png", "*_redraw.png"]
        if engine != "multi_series"
        else ["*_redraw.png"]
    )
    return {
        "xlsx": str(find_artifact(attempt_dir, xlsx_patterns) or ""),
        "overlay": str(find_artifact(attempt_dir, overlay_patterns) or ""),
        "redrawn": str(find_artifact(attempt_dir, redrawn_patterns) or ""),
    }


def contract_snapshot(target: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "confirmed_by_user",
        "confirmation_source",
        "user_confirmation_text",
        "target_selection_note",
        "target_curves",
        "independent_axis",
        "dependent_axis",
        "line_style",
        "dashed_handling",
        "data_form",
        "visible_path_user_confirmed",
        "visible_segments_user_confirmed",
        "non_function_path_user_confirmed",
        "same_color_text_user_confirmed",
        "text_occlusion_user_confirmed",
        "continuity_required",
        "curve_form",
        "target_colors",
        "exclude_same_color_text",
        "single_value_axis",
        "target_type",
        "engine",
    ]
    snapshot = {field: target.get(field) for field in fields if field in target}
    script_args = target.get("script_args")
    if isinstance(script_args, dict):
        useful_script_args = {}
        for key in (
            "curve_preset",
            "trace_mode",
            "profile_global_mask",
            "profile_interpolate_gap_rows",
            "color_centers",
            "color_space",
            "max_color_dist",
            "min_chroma",
            "roi",
            "pixel_roi",
            "guide_points",
            "point_guide_tol_y",
            "x_profile_interpolate_gap_px",
            "x_label",
            "y_label",
        ):
            if key in script_args:
                useful_script_args[key] = script_args.get(key)
        if useful_script_args:
            snapshot["script_args"] = useful_script_args
    return snapshot


def audit_attempt(
    target: dict[str, Any],
    target_type: str,
    engine: str,
    attempt_dir: Path,
    returncode: int,
    input_path: Path | None = None,
    preflight_failures: list[dict[str, Any]] | None = None,
    initial_warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    audit_cfg = dict(target.get("audit") or {})
    failures = list(preflight_failures or [])
    warnings: list[dict[str, str]] = list(initial_warnings or [])
    preflight_only = bool(preflight_failures)
    metrics = parse_reports(attempt_dir)
    points = read_csv_points(attempt_dir)
    metrics["point_count"] = int(len(points))
    metrics["artifacts"] = collect_artifacts(attempt_dir, engine)

    if preflight_only:
        remediation = build_remediation(failures, metrics, reactable=False)
        result = {
            "target_key": target.get("key", ""),
            "target_name": target.get("name", ""),
            "target_type": target_type,
            "engine": engine,
            "effective_contract": contract_snapshot(target),
            "attempt_dir": str(attempt_dir),
            "status": "FAIL",
            "reactable": False,
            "failures": failures,
            "warnings": warnings,
            "metrics": metrics,
            "remediation": remediation,
        }
        write_json(attempt_dir / "audit.json", result)
        write_attempt_summary(attempt_dir / "audit_summary.md", result)
        return result

    if returncode != 0:
        failures.append(failure("script_exit_nonzero", f"extraction command exited with code {returncode}", reactable=False))

    min_points = to_int(audit_cfg.get("min_point_count", 3), 3)
    if len(points) < min_points:
        failures.append(failure("too_few_points", f"only {len(points)} extracted pixel points; minimum is {min_points}", reactable=False))

    hits = forbidden_hits(points, list(target.get("forbidden_regions") or []))
    metrics["forbidden_hits"] = hits
    if hits and not bool_opt(target, audit_cfg, "allow_forbidden_hits", False):
        count = sum(item["hit_count"] for item in hits)
        failures.append(failure("forbidden_region_hit", f"{count} extracted points enter forbidden text/legend/axis regions", reactable=False))

    single_value_axis = str(target.get("single_value_axis", "none")).strip().lower()
    if single_value_axis in {"x_to_y", "y_to_x"}:
        single_value = single_value_axis_metrics(
            attempt_dir,
            single_value_axis,
            tolerance_px=threshold(audit_cfg, "single_value_tolerance_px", 1.5),
        )
        metrics["single_value_axis"] = single_value
        if single_value.get("duplicate_bins", 0) > to_int(audit_cfg.get("max_single_value_duplicate_bins", 0), 0):
            failures.append(
                failure(
                    "single_value_axis_violation",
                    f"{single_value_axis} duplicate_bins={single_value['duplicate_bins']} exceeds 0; max_spread_px={single_value['max_spread_px']:.3f}",
                    reactable=False,
                )
            )

    if target.get("continuity_required") is True:
        if single_value_axis in {"x_to_y", "y_to_x"}:
            continuity_axis = axis_continuity_metrics(
                attempt_dir,
                single_value_axis,
                max_gap_px=threshold(audit_cfg, "max_continuity_axis_gap_px", 20.0),
            )
            metrics["axis_continuity"] = continuity_axis
            if continuity_axis.get("gap_count", 0) > to_int(audit_cfg.get("max_continuity_axis_gaps", 0), 0):
                failures.append(
                    failure(
                        "continuity_axis_gap",
                        f"continuous_curve has {continuity_axis['gap_count']} gap(s) above {continuity_axis['max_allowed_gap_px']:.1f}px; max_gap_px={continuity_axis['max_gap_px']:.3f}",
                        reactable=False,
                    )
                )
        report_continuity = metrics.get("continuity")
        if report_continuity and str(report_continuity.get("status", "")).upper() != "PASS":
            failures.append(
                failure(
                    "continuity_report_check",
                    f"low-level continuity status={report_continuity.get('status')!r}; continuous_curve requires PASS",
                    reactable=False,
                )
            )

    if target_type in PROFILE_TARGET_TYPES:
        profile = metrics.get("profile")
        if not profile:
            failures.append(failure("missing_profile_metrics", "single_depth_profile requires profile metrics in the report", reactable=False))
        else:
            if str(profile.get("status", "")).upper() != "PASS":
                failures.append(
                    failure(
                        "profile_report_check",
                        f"low-level profile report status={profile.get('status')!r}; final accuracy output requires PASS",
                        reactable=True,
                    )
                )
            max_duplicate = 0
            max_gap = 2.0
            min_coverage = 0.90
            if profile["duplicate_depth_rows"] > max_duplicate:
                failures.append(
                    failure(
                        "profile_duplicate_depth_rows",
                        f"duplicate_depth_rows={profile['duplicate_depth_rows']} exceeds {max_duplicate}",
                        reactable=True,
                    )
                )
            if profile["max_row_gap"] > max_gap:
                failures.append(
                    failure("profile_max_row_gap", f"max_row_gap={profile['max_row_gap']} exceeds {max_gap}", reactable=True)
                )
            if profile["row_coverage"] < min_coverage:
                failures.append(
                    failure(
                        "profile_row_coverage",
                        f"row_coverage={profile['row_coverage']:.3f} is below {min_coverage:.3f}",
                        reactable=True,
                    )
                )
            if "max_profile_second_diff_px" in audit_cfg and profile["p95_second_diff"] > threshold(audit_cfg, "max_profile_second_diff_px", float("inf")):
                failures.append(
                    failure(
                        "profile_not_smooth",
                        f"p95_second_diff={profile['p95_second_diff']:.3f}px exceeds max_profile_second_diff_px",
                        reactable=False,
                    )
                )
            if "max_reacted_residual_ratio" in audit_cfg:
                residual_ratio = to_float(metrics.get("last_reacted_residual_ratio", float("nan")))
                if math.isfinite(residual_ratio) and residual_ratio > threshold(audit_cfg, "max_reacted_residual_ratio", 1.0):
                    failures.append(
                        failure(
                            "profile_residual_ratio",
                            f"reacted_residual_ratio={residual_ratio:.4f} exceeds max_reacted_residual_ratio",
                            reactable=True,
                        )
                    )

    if target_type in DASHED_TARGET_TYPES:
        summaries = parse_dashed_summaries(attempt_dir)
        metrics["dashed_summary"] = summaries
        if not summaries:
            failures.append(failure("missing_dashed_summary", "dashed target requires *_endpoint_summary.csv", reactable=False))
        expected_mode = "visible_only" if target_type == "dashed_depth_profile_visible_only" else "continuous"
        for row in summaries:
            if str(row["dash_mode"]) != expected_mode:
                failures.append(
                    failure(
                        "dash_mode_mismatch",
                        f"summary has dash_mode={row['dash_mode']!r}; expected {expected_mode!r}",
                        reactable=False,
                    )
                )
            if row["duplicate_y_rows"] > to_int(audit_cfg.get("max_duplicate_y_rows", 0), 0):
                failures.append(
                    failure(
                        "dashed_duplicate_y_rows",
                        f"{row['series_key']} duplicate_y_rows={row['duplicate_y_rows']} exceeds threshold",
                        reactable=False,
                    )
                )
            if target_type != "dashed_depth_profile_visible_only":
                max_gap = threshold(audit_cfg, "max_y_gap_rows", 2.0)
                if row["max_y_gap_rows"] > max_gap:
                    failures.append(
                        failure(
                            "dashed_max_y_gap_rows",
                            f"{row['series_key']} max_y_gap_rows={row['max_y_gap_rows']} exceeds {max_gap}",
                            reactable=False,
                        )
                    )
            if "max_mask_residual_p95_px" in audit_cfg and row["mask_residual_p95_px"] > threshold(audit_cfg, "max_mask_residual_p95_px", float("inf")):
                failures.append(
                    failure(
                        "dashed_mask_residual_p95",
                        f"{row['series_key']} mask_residual_p95_px={row['mask_residual_p95_px']:.3f} exceeds threshold",
                        reactable=False,
                    )
                )

    if target_type in STRAIGHT_TARGET_TYPES:
        line = line_metrics(points)
        metrics["line_fit"] = line
        if line["rms_px"] > threshold(audit_cfg, "max_line_rms_px", 2.5) or line["p95_px"] > threshold(audit_cfg, "max_line_p95_px", 5.0):
            failures.append(
                failure(
                    "straight_line_residual",
                    f"line RMS={line['rms_px']:.3f}px or p95={line['p95_px']:.3f}px exceeds straight-reference thresholds",
                    reactable=False,
                )
            )

    if target_type in SMOOTH_TARGET_TYPES or bool_opt(target, audit_cfg, "require_smoothness", False):
        smooth = smooth_metrics(points)
        metrics["smoothness"] = smooth
        max_second = threshold(audit_cfg, "max_smooth_p95_second_diff_px", 8.0)
        max_spike_fraction = threshold(audit_cfg, "max_spike_fraction", 0.05)
        max_backtrack = threshold(audit_cfg, "max_backtrack_fraction", 0.20)
        if smooth["p95_second_diff_px"] > max_second:
            failures.append(
                failure(
                    "smooth_curve_jagged",
                    f"p95_second_diff={smooth['p95_second_diff_px']:.3f}px exceeds {max_second}",
                    reactable=False,
                )
            )
        if smooth["spike_fraction"] > max_spike_fraction:
            failures.append(
                failure(
                    "smooth_curve_spikes",
                    f"spike_fraction={smooth['spike_fraction']:.3f} exceeds {max_spike_fraction}",
                    reactable=False,
                )
            )
        if smooth["backtrack_fraction"] > max_backtrack:
            failures.append(
                failure(
                    "smooth_curve_backtrack",
                    f"backtrack_fraction={smooth['backtrack_fraction']:.3f} exceeds {max_backtrack}",
                    reactable=False,
                )
            )

    artifacts = metrics["artifacts"]
    for artifact_key in ("xlsx", "overlay", "redrawn"):
        if not artifacts.get(artifact_key):
            failures.append(failure(f"missing_{artifact_key}", f"required final artifact source {artifact_key!r} is missing", reactable=False))

    if returncode == 0 and input_path is not None and artifacts.get("xlsx"):
        strict_qa = run_strict_qa(target, target_type, engine, input_path, attempt_dir)
        metrics["strict_qa"] = strict_qa
        failures.extend(strict_qa.get("failures") or [])
        warnings.extend(strict_qa.get("warnings") or [])

    status = "PASS" if not failures else "FAIL"
    reactable = bool(failures) and all(item["code"] in REACTABLE_CODES and item["reactable"] for item in failures)
    remediation = build_remediation(failures, metrics, reactable)
    result = {
        "target_key": target.get("key", ""),
        "target_name": target.get("name", ""),
        "target_type": target_type,
        "engine": engine,
        "effective_contract": contract_snapshot(target),
        "attempt_dir": str(attempt_dir),
        "status": status,
        "reactable": reactable,
        "failures": failures,
        "warnings": warnings,
        "metrics": metrics,
        "remediation": remediation,
    }
    write_json(attempt_dir / "audit.json", result)
    write_attempt_summary(attempt_dir / "audit_summary.md", result)
    return result


def safe_remove_dir(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved == root_resolved or root_resolved not in resolved.parents:
        raise RuntimeError(f"Refusing to remove directory outside output root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def publish_final(target_audit: dict[str, Any], out_dir: Path) -> dict[str, str]:
    target_key = slugify(str(target_audit["target_key"]))
    final_dir = out_dir / "final" / target_key
    safe_remove_dir(final_dir, out_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    artifacts = target_audit["metrics"]["artifacts"]
    copies = {
        "result.xlsx": artifacts["xlsx"],
        "overlay.png": artifacts["overlay"],
        "redrawn.png": artifacts["redrawn"],
    }
    final_paths: dict[str, str] = {}
    for final_name, source in copies.items():
        source_path = Path(str(source))
        dest = final_dir / final_name
        shutil.copy2(source_path, dest)
        final_paths[final_name] = str(dest)
    strict_artifacts = ((target_audit.get("metrics") or {}).get("strict_qa") or {}).get("artifacts") or {}
    strict_copies = {
        "strict_qa.json": strict_artifacts.get("json"),
        "strict_qa_summary.md": strict_artifacts.get("summary"),
        "strict_qa_overlay.png": strict_artifacts.get("overlay"),
    }
    for final_name, source in strict_copies.items():
        if not source:
            continue
        source_path = Path(str(source))
        if not source_path.exists():
            continue
        dest = final_dir / final_name
        shutil.copy2(source_path, dest)
        final_paths[final_name] = str(dest)
    target_audit["final_dir"] = str(final_dir)
    target_audit["final_artifacts"] = final_paths
    write_json(final_dir / "audit.json", target_audit)
    write_attempt_summary(final_dir / "audit_summary.md", target_audit)
    return final_paths


def write_attempt_summary(path: Path, audit: dict[str, Any]) -> None:
    remediation = audit.get("remediation") or {}
    lines = [
        f"# Target audit: {audit.get('target_key', '')}",
        "",
        f"Status: {audit.get('status')}",
        f"Target type: {audit.get('target_type')}",
        f"Engine: {audit.get('engine')}",
        f"Attempt dir: {audit.get('attempt_dir')}",
        f"Next action: {remediation.get('next_action', '')}",
        "",
    ]
    failures = audit.get("failures") or []
    if failures:
        lines.append("Failures:")
        for item in failures:
            lines.append(f"- {item['code']}: {item['message']}")
    else:
        lines.append("Failures: none")
    warnings = audit.get("warnings") or []
    if warnings:
        lines.extend(["", "Warnings:"])
        for item in warnings:
            lines.append(f"- {item['code']}: {item['message']}")
    if remediation:
        lines.extend(["", "Remediation:", f"- {remediation.get('summary', '')}"])
        for key, title in (
            ("automatic_fixes", "Automatic fixes to try"),
            ("inspection_steps", "Inspection/refinement steps"),
            ("user_questions", "Questions for the user"),
        ):
            values = remediation.get(key) or []
            if values:
                lines.append(f"- {title}:")
                for value in values:
                    lines.append(f"  - {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_summary(path: Path, case_id: str, overall: dict[str, Any]) -> None:
    lines = [
        f"# Accuracy audit: {case_id}",
        "",
        f"Overall status: {overall['status']}",
        f"Workflow next action: {overall.get('next_action', '')}",
        f"Next steps: {overall.get('next_steps', '')}",
        "",
        "| target | target_type | status | next_action | react_attempts | failures |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for target in overall["targets"]:
        failures = "; ".join(item["code"] for item in target.get("failures", [])) or "none"
        next_action = (target.get("remediation") or {}).get("next_action", "")
        lines.append(
            f"| {target.get('target_key', '')} | {target.get('target_type', '')} | {target.get('status', '')} | {next_action} | "
            f"{target.get('react_attempts', 0)} | {failures} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_steps(path: Path, overall: dict[str, Any]) -> None:
    targets = list(overall.get("targets") or [])
    action = str(overall.get("next_action", overall_next_action(targets)))
    lines = [
        f"# Next steps: {overall.get('case_id', '')}",
        "",
        f"Workflow next action: {action}",
        "",
    ]
    if action == "ready_to_publish":
        lines.append("User-facing response:")
        lines.append("- The digitized outputs are ready. Use the final artifacts listed below.")
        lines.extend(["", "Final artifacts:"])
        for target in targets:
            final_artifacts = target.get("final_artifacts") or {}
            if not final_artifacts:
                continue
            lines.append(f"- {target.get('target_key', '')}:")
            for name in ("result.xlsx", "overlay.png", "redrawn.png", "strict_qa.json", "strict_qa_overlay.png"):
                if name in final_artifacts:
                    lines.append(f"  - {name}: {final_artifacts[name]}")
    elif action == "ask_user":
        questions = collect_unique_remediation_items(targets, "user_questions")
        lines.append("User-facing response:")
        lines.append("- I need a short confirmation before I can produce the final digitized data:")
        for question in questions:
            lines.append(f"  - {question}")
        lines.extend(["", "Agent instruction:", "- Ask these questions directly and wait for the answer before rerunning accuracy mode."])
    elif action == "repair_config_and_rerun":
        fixes = collect_unique_remediation_items(targets, "automatic_fixes")
        lines.append("Agent instruction:")
        lines.append("- Apply these deterministic config fixes and rerun accuracy mode before replying to the user:")
        for fix in fixes:
            lines.append(f"  - {fix}")
    else:
        steps = collect_unique_remediation_items(targets, "inspection_steps")
        questions = collect_unique_remediation_items(targets, "user_questions")
        lines.append("Agent instruction:")
        lines.append("- Continue refining the extraction before treating this as complete:")
        for step in steps:
            lines.append(f"  - {step}")
        if questions:
            lines.extend(["", "Ask the user only if the inspection cannot classify the target safely:"])
            for question in questions:
                lines.append(f"  - {question}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def has_target_colors(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(str(item).strip() for item in value)
    return False


def has_target_curves(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if isinstance(item, dict) and str(item.get("color", "")).strip():
            return True
        if isinstance(item, str) and item.strip():
            return True
    return False


def y_axis_looks_like_depth(target: dict[str, Any]) -> bool:
    script_args = target.get("script_args") if isinstance(target.get("script_args"), dict) else {}
    labels = script_args.get("labels") if isinstance(script_args.get("labels"), dict) else {}
    candidates = [
        script_args.get("y_label"),
        labels.get("y"),
        target.get("target_selection_note"),
        target.get("name"),
    ]
    text = " ".join(str(item) for item in candidates if item)
    return bool(DEPTH_AXIS_RE.search(text))


def derive_contract(target: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Derive internal contract fields from user-facing intent fields."""
    out = dict(target)
    warnings: list[dict[str, str]] = []
    data_form = str(out.get("data_form", "")).strip().lower()
    independent_axis = str(out.get("independent_axis", "")).strip().lower()
    line_style = str(out.get("line_style", "")).strip().lower()

    if data_form in {"visible_segments", "discrete_points", "reference_line"} and independent_axis in {"", "none"}:
        independent_axis = "none"
    if independent_axis:
        out["independent_axis"] = independent_axis

    derived: dict[str, Any] = {}
    if independent_axis == "x":
        derived["dependent_axis"] = "y"
        derived["single_value_axis"] = "x_to_y"
        derived["curve_form"] = "visible_segments" if data_form == "visible_segments" else "normal_xy_x_to_y"
    elif independent_axis == "y":
        derived["dependent_axis"] = "x"
        derived["single_value_axis"] = "y_to_x"
        derived["curve_form"] = "visible_segments" if data_form == "visible_segments" else "depth_profile_y_to_x"
    elif independent_axis == "none":
        derived["dependent_axis"] = "none"
        derived["single_value_axis"] = "none"
        derived["curve_form"] = "visible_segments" if data_form == "visible_segments" else "smooth_path"

    if data_form == "continuous_curve":
        derived["continuity_required"] = True
    elif data_form in {"visible_segments", "discrete_points", "reference_line"}:
        derived["continuity_required"] = False

    if "curve_form" in derived:
        if out.get("curve_form") and out.get("curve_form") != derived["curve_form"]:
            warnings.append(
                warning(
                    "derived_curve_form",
                    f"overrode curve_form={out.get('curve_form')!r} with {derived['curve_form']!r} from user intent",
                )
            )
        out["curve_form"] = derived["curve_form"]
    for key in ("dependent_axis", "single_value_axis", "continuity_required"):
        if key in derived:
            if key in out and out.get(key) != derived[key]:
                warnings.append(warning(f"derived_{key}", f"overrode {key}={out.get(key)!r} with {derived[key]!r} from user intent"))
            out[key] = derived[key]

    curve_form = str(out.get("curve_form", "")).strip()
    current_target_type = str(out.get("target_type", "")).strip()
    derived_target_type = ""
    if curve_form == "depth_profile_y_to_x":
        raw_engine = str(out.get("engine", "")).strip().lower()
        if current_target_type == "multi_series_depth_profile" or raw_engine in {"multi", "multi_series"}:
            derived_target_type = "multi_series_depth_profile"
        else:
            derived_target_type = "single_depth_profile"
    elif data_form == "reference_line":
        derived_target_type = "straight_dashed_reference" if line_style == "dashed" else "straight_reference"
    elif data_form == "discrete_points":
        derived_target_type = "marker_series"
    elif curve_form == "normal_xy_x_to_y":
        compatible_types = {"single_xy_curve", "smooth_xy_curve", "multi_series_xy_curve", "multi_series_smooth_curve_only", "curve_with_markers"}
        if not current_target_type or current_target_type not in compatible_types:
            raw_engine = str(out.get("engine", "")).strip().lower()
            derived_target_type = "multi_series_xy_curve" if raw_engine in {"multi", "multi_series"} else "single_xy_curve"
    elif curve_form in {"smooth_path", "visible_segments"}:
        incompatible_types = PROFILE_TARGET_TYPES | STRAIGHT_TARGET_TYPES | {"marker_series"}
        if not current_target_type or current_target_type in incompatible_types:
            derived_target_type = "visible_polyline"
    elif not current_target_type:
        derived_target_type = "visible_polyline"
    if derived_target_type:
        if out.get("target_type") and out.get("target_type") != derived_target_type:
            warnings.append(
                warning(
                    "derived_target_type",
                    f"overrode target_type={out.get('target_type')!r} with {derived_target_type!r} from user intent",
                )
            )
        out["target_type"] = derived_target_type

    script_args = out.get("script_args")
    if isinstance(script_args, dict) and out.get("curve_form") == "depth_profile_y_to_x":
        script_args = dict(script_args)
        if not script_args.get("trace_mode"):
            script_args["trace_mode"] = "profile"
            warnings.append(warning("derived_trace_mode", "set trace_mode=profile for depth_profile_y_to_x"))
        if line_style == "solid" and script_args.get("profile_global_mask") is None:
            script_args["profile_global_mask"] = True
            warnings.append(warning("derived_profile_global_mask", "set profile_global_mask=true for solid depth profile"))
        out["script_args"] = script_args

    return out, warnings


def validate_confirmation(target: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if target.get("confirmed_by_user") is not True:
        failures.append(failure("missing_confirmation", "accuracy mode requires confirmed_by_user=true before extraction", reactable=False))
    confirmation_source = str(target.get("confirmation_source", "")).strip().lower()
    if not confirmation_source:
        failures.append(
            failure(
                "missing_confirmation_source",
                "record confirmation_source as explicit_user_response or user_delegated_inference before extraction",
                reactable=False,
            )
        )
    elif confirmation_source not in CONFIRMATION_SOURCES:
        failures.append(
            failure(
                "invalid_confirmation_source",
                f"confirmation_source={confirmation_source!r} is not allowed; use explicit_user_response or user_delegated_inference",
                reactable=False,
            )
        )
    if not str(target.get("user_confirmation_text", "")).strip():
        failures.append(
            failure(
                "missing_user_confirmation_text",
                "record the user's target-confirmation answer or explicit delegation text before extraction",
                reactable=False,
            )
        )
    if not str(target.get("target_selection_note", "")).strip():
        failures.append(failure("missing_target_selection_note", "record which visible curve(s) are being digitized", reactable=False))
    if not has_target_curves(target.get("target_curves")):
        failures.append(failure("missing_target_curves", "record target_curves with curve/series names and colors", reactable=False))

    independent_axis = str(target.get("independent_axis", "")).strip().lower()
    dependent_axis = str(target.get("dependent_axis", "")).strip().lower()
    if independent_axis not in INDEPENDENT_AXES:
        failures.append(failure("missing_independent_axis", "independent_axis must be x, y, or none", reactable=False))
    if dependent_axis not in DEPENDENT_AXES:
        failures.append(failure("missing_dependent_axis", "dependent_axis must be x, y, or none", reactable=False))
    if independent_axis in {"x", "y"} and dependent_axis in {"x", "y"} and independent_axis == dependent_axis:
        failures.append(failure("axis_mapping_mismatch", "independent_axis and dependent_axis must be different", reactable=False))
    if independent_axis == "none" and dependent_axis != "none":
        failures.append(failure("axis_mapping_mismatch", "independent_axis=none requires dependent_axis=none", reactable=False))
    if independent_axis in {"x", "y"} and dependent_axis == "none":
        failures.append(failure("axis_mapping_mismatch", f"independent_axis={independent_axis} requires a dependent x/y axis", reactable=False))
    if y_axis_looks_like_depth(target) and independent_axis == "x" and target.get("depth_axis_user_confirmed") is not True:
        failures.append(
            failure(
                "depth_axis_confirmation_required",
                "y-axis label/target note looks like a depth profile; confirm whether y/depth is the independent variable before using independent_axis=x",
                reactable=False,
            )
        )

    line_style = str(target.get("line_style", "")).strip().lower()
    if line_style not in LINE_STYLES:
        failures.append(failure("missing_line_style", "line_style must be solid, dashed, dotted, marker, or mixed_line_marker", reactable=False))
    data_form = str(target.get("data_form", "")).strip().lower()
    if data_form not in DATA_FORMS:
        failures.append(failure("missing_data_form", "data_form must be continuous_curve, visible_segments, discrete_points, or reference_line", reactable=False))
    if "continuity_required" not in target or not isinstance(target.get("continuity_required"), bool):
        failures.append(failure("contract_derivation_failed", "could not derive continuity_required from data_form", reactable=False))
    elif data_form == "continuous_curve" and target.get("continuity_required") is not True:
        failures.append(failure("data_form_continuity_mismatch", "data_form=continuous_curve requires continuity_required=true", reactable=False))
    elif data_form in {"visible_segments", "discrete_points"} and target.get("continuity_required") is not False:
        failures.append(failure("data_form_continuity_mismatch", f"data_form={data_form} requires continuity_required=false", reactable=False))
    if line_style == "dashed" and str(target.get("dashed_handling", "")).strip().lower() not in {"continuous", "visible_only"}:
        failures.append(
            failure(
                "dashed_mode_needs_confirmation",
                "line_style=dashed requires dashed_handling=continuous or visible_only before running accuracy mode",
                reactable=False,
            )
        )
    dashed_visible_only = line_style == "dashed" and str(target.get("dashed_handling", "")).strip().lower() == "visible_only"
    visible_path_confirmed = any(
        target.get(field) is True
        for field in (
            "visible_path_user_confirmed",
            "visible_segments_user_confirmed",
            "non_function_path_user_confirmed",
        )
    )
    visible_path_reasons: list[str] = []
    if data_form == "visible_segments":
        visible_path_reasons.append("data_form=visible_segments is an exception path")
    if independent_axis == "none" and data_form not in {"discrete_points", "reference_line"}:
        visible_path_reasons.append("independent_axis=none disables x/y single-valued QA")
    if y_axis_looks_like_depth(target) and data_form == "visible_segments":
        visible_path_reasons.append("depth-looking targets default to continuous y->x profiles")
    if visible_path_reasons and not (visible_path_confirmed or dashed_visible_only):
        failures.append(
            failure(
                "visible_path_confirmation_required",
                "; ".join(visible_path_reasons) + "; explicitly confirm visible-path extraction before running accuracy mode",
                reactable=False,
            )
        )

    curve_form = str(target.get("curve_form", "")).strip()
    if curve_form not in CURVE_FORMS:
        failures.append(
            failure(
                "contract_derivation_failed",
                "could not derive curve_form from the declared data relationship",
                reactable=False,
            )
        )
    if not has_target_colors(target.get("target_colors")):
        failures.append(failure("missing_target_colors", "record the target color(s) selected by the user", reactable=False))
    if "exclude_same_color_text" not in target or not isinstance(target.get("exclude_same_color_text"), bool):
        failures.append(
            failure(
                "missing_exclude_same_color_text",
                "record exclude_same_color_text as true or false before running accuracy mode",
                reactable=False,
            )
        )
    single_value_axis = str(target.get("single_value_axis", "")).strip()
    if single_value_axis not in SINGLE_VALUE_AXES:
        failures.append(failure("contract_derivation_failed", "could not derive single_value_axis from the declared data relationship", reactable=False))
    expected_axis = {"x": "x_to_y", "y": "y_to_x", "none": "none"}.get(independent_axis)
    if expected_axis and single_value_axis != expected_axis:
        failures.append(
            failure(
                "axis_mapping_mismatch",
                f"independent_axis={independent_axis} requires single_value_axis={expected_axis}, got {single_value_axis!r}",
                reactable=False,
            )
        )
    if independent_axis == "x" and curve_form == "depth_profile_y_to_x":
        failures.append(
            failure(
                "curve_form_axis_mismatch",
                "independent_axis=x cannot use curve_form=depth_profile_y_to_x",
                reactable=False,
            )
        )
    if independent_axis == "y" and curve_form not in {"depth_profile_y_to_x", "visible_segments"}:
        failures.append(
            failure(
                "curve_form_axis_mismatch",
                f"independent_axis=y requires a y-to-x compatible curve_form, got {curve_form!r}",
                reactable=False,
            )
        )
    if independent_axis == "none" and curve_form not in {"smooth_path", "visible_segments"}:
        failures.append(
            failure(
                "curve_form_axis_mismatch",
                f"independent_axis=none requires curve_form=smooth_path or visible_segments, got {curve_form!r}",
                reactable=False,
            )
        )
    return failures


def validate_profile_contract(target: dict[str, Any], target_type: str, engine: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    curve_form = str(target.get("curve_form", "")).strip()
    if curve_form != "depth_profile_y_to_x":
        return failures
    if target_type == "multi_series_depth_profile" and engine == "multi_series":
        return failures
    if target_type != "single_depth_profile":
        failures.append(
            failure(
                "curve_form_target_type_mismatch",
                "depth_profile_y_to_x requires target_type=single_depth_profile or multi_series_depth_profile",
                reactable=False,
            )
        )
    if engine != "single_curve":
        failures.append(failure("curve_form_engine_mismatch", "depth_profile_y_to_x requires engine=single_curve", reactable=False))
    trace_mode = str((target.get("script_args") or {}).get("trace_mode", "")).strip()
    if trace_mode != "profile":
        failures.append(
            failure(
                "depth_profile_requires_profile_mode",
                f"depth_profile_y_to_x requires script_args.trace_mode='profile', got {trace_mode!r}",
                reactable=False,
            )
        )
    return failures


def validate_profile_thresholds(target: dict[str, Any], target_type: str) -> list[dict[str, Any]]:
    if target_type not in PROFILE_TARGET_TYPES:
        return []
    failures: list[dict[str, Any]] = []
    audit = dict(target.get("audit") or {})
    if to_int(audit.get("max_duplicate_depth_rows", 0), 0) > 0:
        failures.append(
            failure(
                "unsafe_threshold_override",
                "accuracy mode does not allow max_duplicate_depth_rows above 0 for single_depth_profile",
                reactable=False,
            )
        )
    if "max_profile_gap_rows" in audit and threshold(audit, "max_profile_gap_rows", 2.0) > 2.0:
        failures.append(
            failure(
                "unsafe_threshold_override",
                "accuracy mode does not allow max_profile_gap_rows above 2 for single_depth_profile",
                reactable=False,
            )
        )
    if "min_profile_row_coverage" in audit and threshold(audit, "min_profile_row_coverage", 0.90) < 0.90:
        failures.append(
            failure(
                "unsafe_threshold_override",
                "accuracy mode does not allow min_profile_row_coverage below 0.90 for single_depth_profile",
                reactable=False,
            )
        )
    return failures


def validate_target(target: dict[str, Any], target_type: str, engine: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    failures.extend(validate_confirmation(target))
    if target_type not in SUPPORTED_TARGET_TYPES:
        failures.append(failure("unsupported_target_type", f"unsupported target_type={target_type!r}", reactable=False))
    if target_type in DASHED_TARGET_TYPES and engine != "dashed_endpoint":
        failures.append(failure("wrong_engine_for_dashed", f"{target_type} requires engine=dashed_endpoint", reactable=False))
    if target_type in {"multi_series_xy_curve", "multi_series_smooth_curve_only"} and engine != "multi_series":
        failures.append(failure("wrong_engine_for_multi_series", f"{target_type} requires engine=multi_series", reactable=False))
    if target_type == "multi_series_depth_profile" and engine != "multi_series":
        failures.append(failure("wrong_engine_for_multi_series_depth_profile", "multi_series_depth_profile requires engine=multi_series", reactable=False))
    if target_type in PROFILE_TARGET_TYPES and engine != "single_curve":
        failures.append(failure("wrong_engine_for_profile", f"{target_type} requires engine=single_curve", reactable=False))
    failures.extend(validate_profile_contract(target, target_type, engine))
    failures.extend(validate_profile_thresholds(target, target_type))
    return failures


def can_auto_repair_preflight(preflight: list[dict[str, Any]]) -> bool:
    if not preflight:
        return False
    codes = failure_codes(preflight)
    if codes & USER_CONFIRMATION_CODES:
        return False
    return bool(codes) and codes <= AUTO_REPAIR_PREFLIGHT_CODES


def auto_repair_target(target: dict[str, Any], preflight: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    repaired = dict(target)
    repaired["script_args"] = dict(target.get("script_args") or {})
    repaired["audit"] = dict(target.get("audit") or {})
    codes = failure_codes(preflight)
    warnings: list[dict[str, str]] = []
    if "depth_profile_requires_profile_mode" in codes:
        repaired["script_args"]["trace_mode"] = "profile"
        warnings.append(warning("auto_repaired_trace_mode", "set script_args.trace_mode=profile for depth_profile_y_to_x"))
    if "unsafe_threshold_override" in codes:
        removed: list[str] = []
        for key in ("max_duplicate_depth_rows", "max_profile_gap_rows", "min_profile_row_coverage"):
            if key in repaired["audit"]:
                repaired["audit"].pop(key, None)
                removed.append(key)
        if removed:
            warnings.append(warning("auto_repaired_profile_thresholds", "removed relaxed depth-profile audit thresholds: " + ", ".join(removed)))
    repaired["auto_repaired_from_failures"] = sorted(codes)
    return repaired, warnings


def run_target(
    target: dict[str, Any],
    input_path: Path,
    out_dir: Path,
    config_base: Path,
    max_react_passes: int,
) -> dict[str, Any]:
    target_key = slugify(str(target.get("key") or target.get("name") or "target"), "target")
    target["key"] = target_key
    target, contract_warnings = derive_contract(target)
    target_type = str(target.get("target_type", "")).strip()
    engine = infer_engine(target, target_type)
    target_dir = out_dir / "attempts" / target_key
    safe_remove_dir(target_dir, out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    preflight = validate_target(target, target_type, engine)
    initial_warnings: list[dict[str, str]] = list(contract_warnings)
    attempt_name = "attempt_0"
    if can_auto_repair_preflight(preflight):
        repaired_target, repair_warnings = auto_repair_target(target, preflight)
        repaired_type = str(repaired_target.get("target_type", "")).strip()
        repaired_engine = infer_engine(repaired_target, repaired_type)
        repaired_preflight = validate_target(repaired_target, repaired_type, repaired_engine)
        if not repaired_preflight:
            target = repaired_target
            target_type = repaired_type
            engine = repaired_engine
            preflight = []
            initial_warnings.extend(repair_warnings)
            attempt_name = "auto_repair_1"
        else:
            initial_warnings.extend(repair_warnings)
            preflight = repaired_preflight
    config_path: Path | None = None
    if engine in {"dashed_endpoint", "multi_series"} and not preflight:
        try:
            config_path = materialize_engine_config(target, target_dir, config_base)
        except ValueError as exc:
            preflight.append(failure("missing_engine_config", str(exc), reactable=False))
        if config_path and engine == "dashed_endpoint":
            preflight.extend(validate_dash_mode(config_path, target_type))

    if preflight:
        attempt_dir = target_dir / "preflight"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        audit = audit_attempt(
            target,
            target_type,
            engine,
            attempt_dir,
            returncode=1,
            input_path=input_path,
            preflight_failures=preflight,
            initial_warnings=initial_warnings,
        )
        audit["react_attempts"] = 0
        final_dir = out_dir / "final" / target_key
        safe_remove_dir(final_dir, out_dir)
        return audit

    attempt_dir = target_dir / attempt_name
    if engine == "single_curve":
        cmd = build_single_curve_command(input_path, attempt_dir, target, react_passes=None)
    else:
        assert config_path is not None
        cmd = build_config_engine_command(engine, input_path, attempt_dir, config_path)
    returncode = run_command(cmd, attempt_dir)
    final_audit = audit_attempt(target, target_type, engine, attempt_dir, returncode, input_path=input_path, initial_warnings=initial_warnings)
    final_audit["react_attempts"] = 0

    if final_audit["status"] != "PASS" and final_audit["reactable"] and target_type in PROFILE_TARGET_TYPES and engine == "single_curve":
        capped = max(0, min(2, max_react_passes))
        for pass_count in range(1, capped + 1):
            react_dir = target_dir / f"react_{pass_count}"
            cmd = build_single_curve_command(input_path, react_dir, target, react_passes=pass_count)
            returncode = run_command(cmd, react_dir)
            react_audit = audit_attempt(target, target_type, engine, react_dir, returncode, input_path=input_path)
            react_audit["react_attempts"] = pass_count
            final_audit = react_audit
            if react_audit["status"] == "PASS" or not react_audit["reactable"]:
                break
        if final_audit["status"] != "PASS" and final_audit.get("react_attempts", 0) >= capped:
            refresh_attempt_remediation(final_audit, reactable=False)

    final_dir = out_dir / "final" / target_key
    if final_audit["status"] == "PASS":
        publish_final(final_audit, out_dir)
    else:
        safe_remove_dir(final_dir, out_dir)
    return final_audit


def run_accuracy(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    config_base = config_path.parent
    case_id = slugify(str(config.get("case_id") or config_path.stem), "case")
    input_path = resolve_path(config["input"], config_base)
    out_dir = resolve_path(config.get("out_dir") or config_base, config_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_react_passes = max(0, min(2, to_int(config.get("max_react_passes", 2), 2)))

    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("accuracy config must contain a non-empty targets list")

    target_audits = []
    for raw_target in targets:
        target = dict(raw_target)
        for field in CONFIRMATION_FIELDS:
            if field not in target and field in config:
                target[field] = config[field]
        target_audits.append(run_target(target, input_path, out_dir, config_base, max_react_passes))
    overall_status = "PASS" if all(item["status"] == "PASS" for item in target_audits) else "FAIL"
    overall = {
        "case_id": case_id,
        "mode": config.get("mode", "accuracy"),
        "precision_mode": config.get("precision_mode", "unspecified"),
        "input": str(input_path),
        "out_dir": str(out_dir),
        "status": overall_status,
        "next_action": overall_next_action(target_audits),
        "next_steps": str(out_dir / "next_steps.md"),
        "targets": target_audits,
    }
    write_next_steps(out_dir / "next_steps.md", overall)
    write_json(out_dir / "audit.json", overall)
    write_root_summary(out_dir / "audit_summary.md", case_id, overall)
    return overall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run accuracy-mode curve digitization with forced audit/react/final gating.")
    parser.add_argument("--config", required=True, help="Fresh accuracy case config JSON.")
    parser.add_argument("--strict-exit", action="store_true", help="Exit with code 1 when the overall audit status is not PASS.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    overall = run_accuracy(config_path)
    print(overall["status"])
    print(Path(overall["out_dir"]) / "audit_summary.md")
    if args.strict_exit and overall["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
