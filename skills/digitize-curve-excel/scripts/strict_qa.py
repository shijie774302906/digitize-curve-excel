from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from digitize_curve_excel import Calibration, apply_dash_bridge, apply_exclude_rects, curve_mask, detect_axes


CURVE_FORMS = {
    "depth_profile_y_to_x",
    "normal_xy_x_to_y",
    "smooth_path",
    "visible_segments",
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def failure(code: str, message: str) -> dict[str, Any]:
    return {"code": code, "message": message, "reactable": False}


def warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rect_list(script_args: dict[str, Any]) -> list[list[float]]:
    value = script_args.get("exclude_rects", script_args.get("exclude_rect", []))
    if not value:
        return []
    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (list, tuple)):
        value = [value]
    rects: list[list[float]] = []
    for rect in value:
        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            rects.append([to_float(part) for part in rect])
    return rects


def triple_list(script_args: dict[str, Any], *keys: str) -> list[list[float]]:
    value: Any = None
    for key in keys:
        if key in script_args:
            value = script_args.get(key)
            break
    if not value:
        return []
    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (list, tuple)):
        value = [value]
    triples: list[list[float]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            triples.append([to_float(part) for part in item])
    return triples


def args_from_script_args(script_args: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        axes=script_args.get("axes"),
        x_min=to_float(script_args.get("x_min"), 0.0),
        x_max=to_float(script_args.get("x_max"), 1.0),
        y_min=to_float(script_args.get("y_min"), 0.0),
        y_max=to_float(script_args.get("y_max"), 1.0),
        reverse_y=bool(script_args.get("reverse_y", False)),
        frame_threshold=to_float(script_args.get("frame_threshold"), 70.0),
        frame_column_fraction=to_float(script_args.get("frame_column_fraction"), 0.30),
        frame_row_fraction=to_float(script_args.get("frame_row_fraction"), 0.25),
        curve_preset=str(script_args.get("curve_preset", "")),
        color_center=triple_list(script_args, "color_centers", "color_center", "centers"),
        color_space=str(script_args.get("color_space", "rgb")).lower(),
        max_color_dist=to_float(script_args.get("max_color_dist", script_args.get("max_dist")), 60.0),
        min_chroma=to_float(script_args.get("min_chroma"), 18.0),
        roi=script_args.get("roi"),
        pixel_roi=script_args.get("pixel_roi"),
        exclude_rect=rect_list(script_args),
        dash_bridge=bool(script_args.get("dash_bridge", False)),
        dash_bridge_x_px=to_float(script_args.get("dash_bridge_x_px"), 9.0),
        dash_bridge_y_px=to_float(script_args.get("dash_bridge_y_px"), 25.0),
        dash_bridge_iterations=to_int(script_args.get("dash_bridge_iterations"), 1),
        react_render_width_px=to_int(script_args.get("react_render_width_px"), 5),
        react_dilation_iterations=to_int(script_args.get("react_dilation_iterations"), 1),
    )


def load_mask(input_path: Path, target: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Calibration, tuple[int, int], np.ndarray]:
    script_args = dict(target.get("script_args") or {})
    args = args_from_script_args(script_args)
    if not args.curve_preset and not args.color_center:
        raise ValueError("single_curve strict QA requires script_args.curve_preset or custom color_centers")
    rgb = np.array(Image.open(input_path).convert("RGB"))
    calib = detect_axes(rgb, args)
    x0, y0, x1, y1 = calib.crop_box
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
    raw_mask = curve_mask(crop, calib, args)
    excluded_mask = apply_exclude_rects(raw_mask, calib, args)
    working_mask = apply_dash_bridge(excluded_mask, args)
    return rgb, raw_mask, calib, (x0, y0), working_mask


def read_digitized_points(attempt_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = sorted(attempt_dir.glob("*_digitized.csv"))
    if not candidates:
        candidates = sorted(path for path in attempt_dir.rglob("*.csv") if path.name.endswith("_digitized.csv"))
    for csv_path in candidates:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "pixel_x" not in reader.fieldnames or "pixel_y" not in reader.fieldnames:
                continue
            for index, row in enumerate(reader, start=1):
                px = to_float(row.get("pixel_x"))
                py = to_float(row.get("pixel_y"))
                if not math.isfinite(px) or not math.isfinite(py):
                    continue
                rows.append(
                    {
                        "pixel_x": px,
                        "pixel_y": py,
                        "point_order": to_int(row.get("point_order"), index),
                        "segment_id": to_int(row.get("segment_id"), 1),
                        "source_csv": str(csv_path),
                    }
                )
    rows.sort(key=lambda item: (item["segment_id"], item["point_order"]))
    return rows


def cluster_indices(indices: np.ndarray, max_gap: int = 1) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    clusters: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        current = int(value)
        if current <= clusters[-1][-1] + max_gap:
            clusters[-1].append(current)
        else:
            clusters.append([current])
    return [np.array(cluster, dtype=int) for cluster in clusters]


def max_consecutive_span(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(set(int(value) for value in values))
    best = 1
    current = 1
    for prev, value in zip(ordered, ordered[1:]):
        if value == prev + 1:
            current += 1
        else:
            best = max(best, current)
            current = 1
    return max(best, current)


def render_points_mask(mask_shape: tuple[int, int], origin: tuple[int, int], points: list[dict[str, Any]], width: int = 5) -> np.ndarray:
    x0, y0 = origin
    render = np.zeros(mask_shape, dtype=np.uint8)
    by_segment: dict[int, list[tuple[int, int]]] = {}
    for point in points:
        local_x = int(round(float(point["pixel_x"]) - x0))
        local_y = int(round(float(point["pixel_y"]) - y0))
        if 0 <= local_x < mask_shape[1] and 0 <= local_y < mask_shape[0]:
            by_segment.setdefault(int(point.get("segment_id", 1)), []).append((local_x, local_y))
    for segment_points in by_segment.values():
        arr = np.array(segment_points, dtype=np.int32)
        if len(arr) > 1:
            cv2.polylines(render, [arr], isClosed=False, color=255, thickness=max(1, width), lineType=cv2.LINE_AA)
        elif len(arr) == 1:
            cv2.circle(render, tuple(arr[0]), radius=max(1, width // 2), color=255, thickness=-1)
    return render > 0


def component_rows(mask: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    rows: list[dict[str, Any]] = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(labels == label)
        rows.append(
            {
                "label": label,
                "area": area,
                "left": int(xs.min()),
                "right": int(xs.max()),
                "top": int(ys.min()),
                "bottom": int(ys.max()),
                "height": int(ys.max() - ys.min() + 1),
                "width": int(xs.max() - xs.min() + 1),
                "mask": labels == label,
            }
        )
    return rows


def point_rows(points: list[dict[str, Any]], origin: tuple[int, int], height: int) -> dict[int, list[float]]:
    x0, y0 = origin
    rows: dict[int, list[float]] = {}
    for point in points:
        row = int(round(float(point["pixel_y"]) - y0))
        x = float(point["pixel_x"]) - x0
        if 0 <= row < height:
            rows.setdefault(row, []).append(x)
    return rows


def raw_exclusion_metrics(raw_mask: np.ndarray, working_mask: np.ndarray) -> dict[str, Any]:
    excluded = raw_mask & ~working_mask
    excluded_rows = np.where(excluded.any(axis=1))[0].astype(int).tolist()
    working_rows = np.where(working_mask.any(axis=1))[0].astype(int).tolist()
    raw_rows = np.where(raw_mask.any(axis=1))[0].astype(int).tolist()
    excluded_only_rows = [row for row in excluded_rows if not working_mask[row].any()]
    return {
        "raw_mask_pixels": int(raw_mask.sum()),
        "working_mask_pixels": int(working_mask.sum()),
        "excluded_target_pixels": int(excluded.sum()),
        "excluded_target_pixel_fraction": float(excluded.sum() / max(raw_mask.sum(), 1)),
        "raw_mask_rows": int(len(raw_rows)),
        "working_mask_rows": int(len(working_rows)),
        "excluded_target_rows": int(len(excluded_rows)),
        "max_excluded_only_row_span": int(max_consecutive_span(excluded_only_rows)),
    }


def row_run_depth_profile_qa(
    target: dict[str, Any],
    raw_mask: np.ndarray,
    working_mask: np.ndarray,
    origin: tuple[int, int],
    points: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    audit = dict(target.get("audit") or {})
    tolerance_px = to_float(audit.get("strict_row_run_tolerance_px"), 2.5)
    max_missing = to_int(audit.get("strict_max_missing_mask_rows"), 0)
    max_duplicate = to_int(audit.get("strict_max_duplicate_mask_rows"), 0)
    configured_max_extra_run_rows = audit.get("strict_max_extra_run_rows")
    max_outside = to_int(audit.get("strict_max_row_rep_outside_run"), 0)
    min_row_coverage = to_float(audit.get("strict_min_row_run_coverage"), 0.98)
    max_excluded_span = to_int(audit.get("strict_max_excluded_only_row_span"), 80)
    min_component_area = to_int(audit.get("strict_component_min_area"), 25)
    script_args = target.get("script_args") if isinstance(target.get("script_args"), dict) else {}
    default_allow_collapsed_runs = (
        str(target.get("curve_form", "")).strip() == "depth_profile_y_to_x"
        and str(target.get("data_form", "")).strip() == "continuous_curve"
        and bool(script_args.get("profile_global_mask", False))
    )
    allow_collapsed_runs = bool(audit.get("strict_allow_collapsed_extra_run_components", default_allow_collapsed_runs))
    collapsed_max_height = to_int(audit.get("strict_collapsed_run_max_height_px"), 8)
    collapsed_min_aspect = to_float(audit.get("strict_collapsed_run_min_aspect"), 6.0)

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    rows_with_mask = np.where(working_mask.any(axis=1))[0].astype(int).tolist()
    selected_by_row = point_rows(points, origin, working_mask.shape[0])

    missing_rows: list[int] = []
    duplicate_rows: list[int] = []
    outside_rows: list[int] = []
    extra_run_rows: list[int] = []
    selected_rows = 0

    for row in rows_with_mask:
        xs = np.where(working_mask[row])[0]
        clusters = cluster_indices(xs)
        selected = selected_by_row.get(row, [])
        if not selected:
            missing_rows.append(row)
            continue
        if len(selected) > 1:
            duplicate_rows.append(row)
        selected_rows += 1
        representative = float(np.median(selected))
        if not any(float(cluster[0]) - tolerance_px <= representative <= float(cluster[-1]) + tolerance_px for cluster in clusters):
            outside_rows.append(row)
        if len(clusters) > 1:
            extra_run_rows.append(row)

    coverage = selected_rows / max(len(rows_with_mask), 1)
    exclusion = raw_exclusion_metrics(raw_mask, working_mask)

    rendered = render_points_mask(working_mask.shape, origin, points, width=5)
    components = component_rows(working_mask, min_component_area)
    unclassified: list[dict[str, int | float]] = []
    collapsed_extra_runs: list[dict[str, int | float]] = []
    for component in components:
        comp_mask = component["mask"]
        comp_rows = np.where(comp_mask.any(axis=1))[0]
        hit_rows = 0
        for row in comp_rows:
            xs = np.where(comp_mask[row])[0]
            selected = selected_by_row.get(int(row), [])
            if selected and any(xs.min() - tolerance_px <= x <= xs.max() + tolerance_px for x in selected):
                hit_rows += 1
        row_hit_fraction = hit_rows / max(len(comp_rows), 1)
        rendered_overlap = int(np.count_nonzero(comp_mask & rendered))
        if row_hit_fraction < 0.35 and rendered_overlap / max(int(component["area"]), 1) < 0.20:
            rows_are_covered = all(int(row) in selected_by_row for row in comp_rows)
            is_collapsed_extra_run = (
                allow_collapsed_runs
                and rows_are_covered
                and int(component["height"]) <= collapsed_max_height
                and int(component["width"]) >= max(1, int(component["height"])) * collapsed_min_aspect
            )
            if is_collapsed_extra_run:
                collapsed_extra_runs.append(
                    {
                        "area": int(component["area"]),
                        "left": int(component["left"]),
                        "right": int(component["right"]),
                        "top": int(component["top"]),
                        "bottom": int(component["bottom"]),
                        "row_hit_fraction": float(row_hit_fraction),
                    }
                )
                continue
            unclassified.append(
                {
                    "area": int(component["area"]),
                    "left": int(component["left"]),
                    "right": int(component["right"]),
                    "top": int(component["top"]),
                    "bottom": int(component["bottom"]),
                    "row_hit_fraction": float(row_hit_fraction),
                }
            )

    metrics: dict[str, Any] = {
        "curve_form": "depth_profile_y_to_x",
        "mask_rows": int(len(rows_with_mask)),
        "selected_mask_rows": int(selected_rows),
        "row_run_coverage": float(coverage),
        "missing_mask_rows": int(len(missing_rows)),
        "max_consecutive_missing_mask_rows": int(max_consecutive_span(missing_rows)),
        "duplicate_mask_rows": int(len(duplicate_rows)),
        "row_rep_outside_run_count": int(len(outside_rows)),
        "extra_run_rows": int(len(extra_run_rows)),
        "collapsed_extra_run_components": int(len(collapsed_extra_runs)),
        "collapsed_extra_run_component_samples": collapsed_extra_runs[:10],
        "unclassified_target_components": int(len(unclassified)),
        "unclassified_target_component_samples": unclassified[:10],
        **exclusion,
    }
    max_extra_run_rows = (
        to_int(configured_max_extra_run_rows, 0)
        if configured_max_extra_run_rows is not None
        else max(10, int(math.ceil(len(rows_with_mask) * 0.20)))
    )
    metrics["max_extra_run_rows_allowed"] = int(max_extra_run_rows)

    if coverage < min_row_coverage:
        failures.append(failure("strict_depth_row_run_coverage", f"row_run_coverage={coverage:.3f} is below {min_row_coverage:.3f}"))
    if len(missing_rows) > max_missing:
        failures.append(failure("strict_depth_missing_rows", f"missing_mask_rows={len(missing_rows)} exceeds {max_missing}"))
    if max_consecutive_span(missing_rows) > max_missing:
        failures.append(
            failure(
                "strict_depth_missing_span",
                f"max_consecutive_missing_mask_rows={max_consecutive_span(missing_rows)} exceeds {max_missing}",
            )
        )
    if len(duplicate_rows) > max_duplicate:
        failures.append(failure("strict_depth_duplicate_rows", f"duplicate_mask_rows={len(duplicate_rows)} exceeds {max_duplicate}"))
    if len(outside_rows) > max_outside:
        failures.append(failure("strict_depth_representative_outside_run", f"row_rep_outside_run_count={len(outside_rows)} exceeds {max_outside}"))
    if len(extra_run_rows) > max_extra_run_rows:
        failures.append(failure("strict_depth_extra_runs", f"extra_run_rows={len(extra_run_rows)} exceeds {max_extra_run_rows}"))
    if exclusion["max_excluded_only_row_span"] > max_excluded_span:
        failures.append(
            failure(
                "strict_broad_exclusion_data_loss",
                f"max_excluded_only_row_span={exclusion['max_excluded_only_row_span']} exceeds {max_excluded_span}",
            )
        )
    if unclassified and bool(target.get("exclude_same_color_text", False)):
        failures.append(
            failure(
                "strict_unclassified_same_color_component",
                f"{len(unclassified)} target-colored component(s) remain unclassified by extracted rows or exclude regions",
            )
        )
    elif unclassified:
        warnings.append(warning("strict_possible_same_color_component", f"{len(unclassified)} target-colored component(s) are not followed by the extraction"))

    return metrics, failures, warnings


def residual_curve_qa(
    target: dict[str, Any],
    working_mask: np.ndarray,
    origin: tuple[int, int],
    points: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    audit = dict(target.get("audit") or {})
    render_width = to_int(audit.get("strict_render_width_px"), 5)
    max_residual_ratio = to_float(audit.get("strict_max_mask_residual_ratio"), 0.25)
    max_component_area = to_int(audit.get("strict_max_residual_component_area"), 80)
    rendered = render_points_mask(working_mask.shape, origin, points, width=render_width)
    residual = working_mask & ~rendered
    components = component_rows(residual, min_area=1)
    max_area = max((int(component["area"]) for component in components), default=0)
    ratio = float(residual.sum() / max(working_mask.sum(), 1))
    metrics = {
        "mask_pixels": int(working_mask.sum()),
        "covered_mask_pixels": int((working_mask & rendered).sum()),
        "residual_pixels": int(residual.sum()),
        "residual_ratio": ratio,
        "max_residual_component_area": int(max_area),
    }
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    if ratio > max_residual_ratio:
        failures.append(failure("strict_mask_residual_ratio", f"mask residual ratio={ratio:.3f} exceeds {max_residual_ratio:.3f}"))
    if max_area > max_component_area:
        failures.append(failure("strict_mask_residual_component", f"max residual component area={max_area} exceeds {max_component_area}"))
    return metrics, failures, warnings


def write_overlay(
    path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int],
    points: list[dict[str, Any]],
) -> None:
    x0, y0 = origin
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.alpha_composite(base, Image.new("RGBA", base.size, (255, 255, 255, 120)))
    layer = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    ys, xs = np.where(mask)
    layer[ys + y0, xs + x0] = np.array([192, 0, 0, 180], dtype=np.uint8)
    overlay = Image.alpha_composite(overlay, Image.fromarray(layer, mode="RGBA"))
    draw = ImageDraw.Draw(overlay)
    by_segment: dict[int, list[tuple[float, float]]] = {}
    for point in points:
        by_segment.setdefault(int(point.get("segment_id", 1)), []).append((float(point["pixel_x"]), float(point["pixel_y"])))
    for segment_points in by_segment.values():
        if len(segment_points) > 1:
            draw.line(segment_points, fill=(0, 166, 255, 255), width=2, joint="curve")
        for px, py in segment_points:
            draw.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=(255, 215, 0, 230))
    overlay.convert("RGB").save(path)


def write_summary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# Strict QA: {result.get('target_key', '')}",
        "",
        f"Status: {result.get('status')}",
        f"Curve form: {result.get('curve_form')}",
        "",
    ]
    failures = result.get("failures") or []
    if failures:
        lines.append("Failures:")
        for item in failures:
            lines.append(f"- {item['code']}: {item['message']}")
    else:
        lines.append("Failures: none")
    warnings = result.get("warnings") or []
    if warnings:
        lines.extend(["", "Warnings:"])
        for item in warnings:
            lines.append(f"- {item['code']}: {item['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_strict_qa(
    target: dict[str, Any],
    target_type: str,
    engine: str,
    input_path: Path,
    attempt_dir: Path,
) -> dict[str, Any]:
    curve_form = str(target.get("curve_form", "")).strip()
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}

    if curve_form not in CURVE_FORMS:
        failures.append(failure("strict_unknown_curve_form", f"unsupported curve_form={curve_form!r}"))
    points = read_digitized_points(attempt_dir)
    if not points:
        failures.append(failure("strict_missing_digitized_points", "no *_digitized.csv with pixel_x/pixel_y was found"))

    if engine != "single_curve":
        warnings.append(warning("strict_qa_limited_engine", f"strict mask QA is limited for engine={engine}; low-level engine audit still applies"))
        metrics["engine_limited"] = True
    elif not failures:
        try:
            rgb, raw_mask, _calib, origin, working_mask = load_mask(input_path, target)
            if curve_form == "depth_profile_y_to_x":
                metrics, depth_failures, depth_warnings = row_run_depth_profile_qa(target, raw_mask, working_mask, origin, points)
                failures.extend(depth_failures)
                warnings.extend(depth_warnings)
            else:
                metrics, residual_failures, residual_warnings = residual_curve_qa(target, working_mask, origin, points)
                metrics["curve_form"] = curve_form
                failures.extend(residual_failures)
                warnings.extend(residual_warnings)
            overlay_path = attempt_dir / "strict_qa_overlay.png"
            write_overlay(overlay_path, rgb, working_mask, origin, points)
            artifacts["overlay"] = str(overlay_path)
        except Exception as exc:  # pragma: no cover - defensive audit output.
            failures.append(failure("strict_qa_exception", f"{type(exc).__name__}: {exc}"))

    status = "PASS" if not failures else "FAIL"
    result = {
        "target_key": target.get("key", ""),
        "target_name": target.get("name", ""),
        "target_type": target_type,
        "engine": engine,
        "curve_form": curve_form,
        "attempt_dir": str(attempt_dir),
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    json_path = attempt_dir / "strict_qa.json"
    summary_path = attempt_dir / "strict_qa_summary.md"
    write_json(json_path, result)
    write_summary(summary_path, result)
    result["artifacts"]["json"] = str(json_path)
    result["artifacts"]["summary"] = str(summary_path)
    return result
