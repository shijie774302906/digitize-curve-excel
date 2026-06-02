from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Axis:
    left: float
    right: float
    top: float
    bottom: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    reverse_y: bool = True

    def pixel_to_data(self, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.x_min + (px - self.left) / (self.right - self.left) * (self.x_max - self.x_min)
        frac_y = (py - self.top) / (self.bottom - self.top)
        if self.reverse_y:
            y = self.y_min + frac_y * (self.y_max - self.y_min)
        else:
            y = self.y_max - frac_y * (self.y_max - self.y_min)
        return x, y

    def data_to_pixel(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = self.left + (x - self.x_min) / (self.x_max - self.x_min) * (self.right - self.left)
        if self.reverse_y:
            py = self.top + (y - self.y_min) / (self.y_max - self.y_min) * (self.bottom - self.top)
        else:
            py = self.bottom - (y - self.y_min) / (self.y_max - self.y_min) * (self.bottom - self.top)
        return px, py

    def local_to_data(self, lx: np.ndarray, ly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.pixel_to_data(lx + self.left, ly + self.top)


@dataclass(frozen=True)
class SeriesConfig:
    key: str
    name: str
    panel: str
    axis: Axis
    preset: str | None
    color_centers: tuple[tuple[int, int, int], ...]
    max_dist: float
    color_space: str
    min_chroma: float
    line_color: str
    dash_mode: str
    path_order: str
    min_area: int
    close_iterations: int
    exclude_rects: tuple[tuple[int, int, int, int], ...]
    roi: tuple[float, float, float, float] | None
    merge_y_px: float
    merge_x_px: float
    suspicious_gap_y_px: float
    suspicious_gap_x_px: float
    max_horizontal_width_px: int
    max_horizontal_height_px: int


@dataclass
class DashComponent:
    label: int
    area: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    center_x: float
    center_y: float
    top_x: float
    top_y: float
    bottom_x: float
    bottom_y: float
    length_px: float
    dx_dy: float


@dataclass
class SeriesResult:
    config: SeriesConfig
    mask: np.ndarray
    components: list[DashComponent]
    links: list[dict[str, float | int | str]]
    profile: list[dict[str, float | int | str]]
    diagnostics: dict[str, float | int | str]


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return text or "series"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {value!r}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def parse_rects(items: list[Any]) -> tuple[tuple[int, int, int, int], ...]:
    rects: list[tuple[int, int, int, int]] = []
    for item in items:
        rect = item.get("rect", item) if isinstance(item, dict) else item
        if len(rect) != 4:
            raise ValueError(f"Invalid exclude region: {item!r}")
        rects.append(tuple(int(round(float(value))) for value in rect))
    return tuple(rects)


def parse_axis(raw: dict[str, Any], default_reverse_y: bool) -> Axis:
    return Axis(
        left=float(raw["left"]),
        right=float(raw["right"]),
        top=float(raw["top"]),
        bottom=float(raw["bottom"]),
        x_min=float(raw["x_min"]),
        x_max=float(raw["x_max"]),
        y_min=float(raw["y_min"]),
        y_max=float(raw["y_max"]),
        reverse_y=bool(raw.get("reverse_y", default_reverse_y)),
    )


def read_config(path: Path) -> tuple[str, list[SeriesConfig]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    prefix = slugify(str(raw.get("output_prefix", path.stem)))
    default_reverse_y = bool(raw.get("reverse_y", True))
    global_exclude = parse_rects(raw.get("exclude_regions", []))
    series: list[SeriesConfig] = []
    seen: set[str] = set()

    for item in raw["series"]:
        name = str(item["name"])
        key = slugify(str(item.get("key", name)))
        base_key = key
        suffix = 2
        while key in seen:
            key = f"{base_key}_{suffix}"
            suffix += 1
        seen.add(key)

        line_color = str(item.get("line_color", item.get("color", "#000000")))
        centers_raw = item.get("color_centers", item.get("centers"))
        color_centers: tuple[tuple[int, int, int], ...]
        if centers_raw:
            color_centers = tuple(tuple(int(round(float(channel))) for channel in center) for center in centers_raw)
        else:
            color_centers = (hex_to_rgb(line_color),)
        if any(len(center) != 3 for center in color_centers):
            raise ValueError(f"Invalid RGB center for series {name!r}")

        dash_mode = str(item.get("dash_mode", "continuous")).lower()
        if dash_mode in {"endpoint", "endpoint_trace", "connect"}:
            dash_mode = "continuous"
        if dash_mode in {"visible", "visible_endpoints", "no_gap"}:
            dash_mode = "visible_only"
        if dash_mode not in {"continuous", "visible_only"}:
            raise ValueError(f"Unsupported dash_mode={dash_mode!r}; use continuous or visible_only")

        axis_raw = item.get("axes", raw.get("axes"))
        if axis_raw is None:
            raise ValueError(f"Series {name!r} must define axes or use top-level axes")

        series_exclude = global_exclude + parse_rects(item.get("exclude_regions", []))
        roi_raw = item.get("roi")
        roi = tuple(float(value) for value in roi_raw) if roi_raw is not None else None
        if roi is not None and len(roi) != 4:
            raise ValueError(f"Series {name!r} has invalid roi; expected [x0,x1,y0,y1]")

        series.append(
            SeriesConfig(
                key=key,
                name=name,
                panel=str(item.get("panel", "")),
                axis=parse_axis(axis_raw, default_reverse_y),
                preset=str(item.get("curve_preset", item.get("preset"))).lower() if item.get("curve_preset", item.get("preset")) else None,
                color_centers=color_centers,
                max_dist=float(item.get("max_dist", 60.0)),
                color_space=str(item.get("color_space", raw.get("color_space", "rgb"))).lower(),
                min_chroma=float(item.get("min_chroma", raw.get("min_chroma", 18.0))),
                line_color=line_color,
                dash_mode=dash_mode,
                path_order=str(item.get("path_order", "y")).lower(),
                min_area=int(item.get("min_area", 2)),
                close_iterations=int(item.get("close_iterations", 1)),
                exclude_rects=series_exclude,
                roi=roi,  # type: ignore[arg-type]
                merge_y_px=float(item.get("merge_same_dash_y_px", 3.0)),
                merge_x_px=float(item.get("merge_same_dash_x_px", 6.0)),
                suspicious_gap_y_px=float(item.get("suspicious_gap_y_px", 70.0)),
                suspicious_gap_x_px=float(item.get("suspicious_gap_x_px", 42.0)),
                max_horizontal_width_px=int(item.get("max_horizontal_width_px", 46)),
                max_horizontal_height_px=int(item.get("max_horizontal_height_px", 3)),
            )
        )
    return prefix, series


def crop_series(rgb: np.ndarray, axis: Axis) -> np.ndarray:
    return rgb[int(round(axis.top)) : int(round(axis.bottom)) + 1, int(round(axis.left)) : int(round(axis.right)) + 1]


def preset_mask(crop: np.ndarray, preset: str) -> np.ndarray:
    r = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    b = crop[:, :, 2].astype(np.int16)
    gray = crop.mean(axis=2)
    if preset == "red":
        return (r > 95) & (g < 110) & (b < 110) & ((r - g) > 25) & ((r - b) > 25)
    if preset == "blue":
        return (b > 80) & (r < 130) & (g < 150) & ((b - r) > 20)
    if preset == "blue-solid":
        return (b > 130) & (r < 90) & (g < 110) & ((b - g) > 60) & ((b - r) > 80)
    if preset == "green":
        return (g > 80) & (r < 150) & (b < 150) & ((g - r) > 15)
    if preset == "purple":
        return (r > 90) & (b > 90) & (g < 150) & ((r - g) > 18) & ((b - g) > 8)
    if preset == "dark":
        return gray < 95
    raise ValueError(f"Unsupported curve preset: {preset!r}")


def center_distance_mask(crop: np.ndarray, cfg: SeriesConfig) -> np.ndarray:
    arr_rgb = crop.astype(np.float32)
    if cfg.color_space == "lab":
        arr = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
        centers_rgb = np.array(cfg.color_centers, dtype=np.uint8).reshape(-1, 1, 3)
        centers = cv2.cvtColor(centers_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    elif cfg.color_space == "rgb":
        arr = arr_rgb
        centers = np.array(cfg.color_centers, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported color_space={cfg.color_space!r}; use rgb or lab")
    diff = arr[:, :, None, :] - centers[None, None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=3)).min(axis=2)
    chroma = arr_rgb.max(axis=2) - arr_rgb.min(axis=2)
    return (dist <= cfg.max_dist) & (chroma >= cfg.min_chroma)


def apply_exclusions(mask: np.ndarray, cfg: SeriesConfig) -> np.ndarray:
    out = mask.copy()
    for x0, y0, x1, y1 in cfg.exclude_rects:
        lx0 = max(0, min(x0, x1) - int(round(cfg.axis.left)))
        lx1 = min(out.shape[1] - 1, max(x0, x1) - int(round(cfg.axis.left)))
        ly0 = max(0, min(y0, y1) - int(round(cfg.axis.top)))
        ly1 = min(out.shape[0] - 1, max(y0, y1) - int(round(cfg.axis.top)))
        if lx1 >= lx0 and ly1 >= ly0:
            out[ly0 : ly1 + 1, lx0 : lx1 + 1] = False
    return out


def apply_roi(mask: np.ndarray, cfg: SeriesConfig) -> np.ndarray:
    if cfg.roi is None:
        return mask
    x0, x1, y0, y1 = cfg.roi
    px_a, py_a = cfg.axis.data_to_pixel(np.array([x0]), np.array([y0]))
    px_b, py_b = cfg.axis.data_to_pixel(np.array([x1]), np.array([y1]))
    lx0 = max(0, int(np.floor(min(px_a[0], px_b[0]) - cfg.axis.left)))
    lx1 = min(mask.shape[1] - 1, int(np.ceil(max(px_a[0], px_b[0]) - cfg.axis.left)))
    ly0 = max(0, int(np.floor(min(py_a[0], py_b[0]) - cfg.axis.top)))
    ly1 = min(mask.shape[0] - 1, int(np.ceil(max(py_a[0], py_b[0]) - cfg.axis.top)))
    roi_mask = np.zeros_like(mask, dtype=bool)
    if lx1 >= lx0 and ly1 >= ly0:
        roi_mask[ly0 : ly1 + 1, lx0 : lx1 + 1] = True
    return mask & roi_mask


def build_mask(crop: np.ndarray, cfg: SeriesConfig) -> np.ndarray:
    mask = preset_mask(crop, cfg.preset) if cfg.preset else center_distance_mask(crop, cfg)
    mask = apply_roi(mask, cfg)
    mask = apply_exclusions(mask, cfg)
    if cfg.close_iterations > 0:
        mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=cfg.close_iterations) > 0
    return mask


def pca_endpoints(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = points_xy.astype(float)
    if len(pts) == 1:
        return pts[0], pts[0]
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    projection = centered @ axis
    return pts[int(np.argmin(projection))], pts[int(np.argmax(projection))]


def component_from_points(label: int, area: int, bbox: tuple[int, int, int, int], xs: np.ndarray, ys: np.ndarray) -> DashComponent:
    points = np.column_stack([xs.astype(float), ys.astype(float)])
    a, b = pca_endpoints(points)
    top, bottom = (a, b) if a[1] <= b[1] else (b, a)
    dx = float(bottom[0] - top[0])
    dy = float(bottom[1] - top[1])
    return DashComponent(
        label=label,
        area=area,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2],
        bbox_h=bbox[3],
        center_x=float(np.median(xs)),
        center_y=float(np.median(ys)),
        top_x=float(top[0]),
        top_y=float(top[1]),
        bottom_x=float(bottom[0]),
        bottom_y=float(bottom[1]),
        length_px=float(np.hypot(dx, dy)),
        dx_dy=dx / dy if abs(dy) >= 1e-6 else 0.0,
    )


def merge_near_same_dash(comps: list[DashComponent], cfg: SeriesConfig) -> list[DashComponent]:
    if not comps:
        return []
    ordered = sorted(comps, key=lambda item: (item.center_y, item.center_x))
    groups: list[list[DashComponent]] = []
    for comp in ordered:
        if not groups:
            groups.append([comp])
            continue
        prev = groups[-1][-1]
        if abs(comp.center_y - prev.center_y) <= cfg.merge_y_px and abs(comp.center_x - prev.center_x) <= cfg.merge_x_px:
            groups[-1].append(comp)
        else:
            groups.append([comp])

    merged: list[DashComponent] = []
    for label, group in enumerate(groups, start=1):
        if len(group) == 1:
            item = group[0]
            item.label = label
            merged.append(item)
            continue
        xs = np.array([value for item in group for value in (item.top_x, item.bottom_x, item.center_x)], dtype=float)
        ys = np.array([value for item in group for value in (item.top_y, item.bottom_y, item.center_y)], dtype=float)
        x0 = int(min(item.bbox_x for item in group))
        y0 = int(min(item.bbox_y for item in group))
        x1 = int(max(item.bbox_x + item.bbox_w for item in group))
        y1 = int(max(item.bbox_y + item.bbox_h for item in group))
        merged.append(component_from_points(label, sum(item.area for item in group), (x0, y0, x1 - x0, y1 - y0), xs, ys))
    return merged


def extract_components(mask: np.ndarray, cfg: SeriesConfig) -> list[DashComponent]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps: list[DashComponent] = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < cfg.min_area:
            continue
        if w > cfg.max_horizontal_width_px and h <= cfg.max_horizontal_height_px:
            continue
        ys, xs = np.where(labels == label)
        comps.append(component_from_points(label, area, (x, y, w, h), xs, ys))
    comps = merge_near_same_dash(comps, cfg)
    if cfg.path_order == "x":
        comps.sort(key=lambda item: (item.center_x, item.center_y))
    else:
        comps.sort(key=lambda item: (item.center_y, item.center_x))
    for idx, comp in enumerate(comps, start=1):
        comp.label = idx
    return comps


def endpoint_polyline(comps: list[DashComponent], cfg: SeriesConfig) -> tuple[list[tuple[float, float]], list[dict[str, float | int | str]]]:
    points: list[tuple[float, float]] = []
    links: list[dict[str, float | int | str]] = []
    if cfg.dash_mode == "visible_only":
        for comp in comps:
            points.extend([(comp.top_x, comp.top_y), (comp.bottom_x, comp.bottom_y)])
        return points, links

    previous: DashComponent | None = None
    for comp in comps:
        if previous is None:
            points.extend([(comp.top_x, comp.top_y), (comp.bottom_x, comp.bottom_y)])
            previous = comp
            continue
        gap_y = comp.top_y - previous.bottom_y
        gap_x = comp.top_x - previous.bottom_x
        status = "accepted"
        if gap_y < -3:
            status = "overlap_or_backtrack"
        elif gap_y > cfg.suspicious_gap_y_px:
            status = "large_gap"
        elif abs(gap_x) > cfg.suspicious_gap_x_px and gap_y < 18:
            status = "large_lateral_jump"
        links.append(
            {
                "from_label": previous.label,
                "to_label": comp.label,
                "from_bottom_x": previous.bottom_x,
                "from_bottom_y": previous.bottom_y,
                "to_top_x": comp.top_x,
                "to_top_y": comp.top_y,
                "gap_x_px": gap_x,
                "gap_y_px": gap_y,
                "distance_px": float(np.hypot(gap_x, gap_y)),
                "status": status,
            }
        )
        points.extend([(comp.top_x, comp.top_y), (comp.bottom_x, comp.bottom_y)])
        previous = comp
    return points, links


def profile_from_polyline(points: list[tuple[float, float]], cfg: SeriesConfig) -> list[dict[str, float | int | str]]:
    by_y: dict[int, list[float]] = {}
    pairs = zip(points[0::2], points[1::2]) if cfg.dash_mode == "visible_only" else zip(points[:-1], points[1:])
    for a, b in pairs:
        x0, y0 = a
        x1, y1 = b
        if abs(y1 - y0) < 1e-6:
            y = int(round((y0 + y1) * 0.5))
            by_y.setdefault(y, []).append((x0 + x1) * 0.5)
            continue
        if y1 < y0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        start = int(np.ceil(y0))
        end = int(np.floor(y1))
        if end < start:
            y = int(round((y0 + y1) * 0.5))
            by_y.setdefault(y, []).append(x0 + 0.5 * (x1 - x0))
            continue
        for y in range(start, end + 1):
            by_y.setdefault(y, []).append(x0 + (y - y0) / (y1 - y0) * (x1 - x0))

    rows: list[dict[str, float | int | str]] = []
    for order, y in enumerate(sorted(by_y), start=1):
        values = np.array(by_y[y], dtype=float)
        x = float(np.median(values))
        x_data, y_data = cfg.axis.local_to_data(np.array([x]), np.array([float(y)]))
        rows.append(
            {
                "series_key": cfg.key,
                "series_name": cfg.name,
                "panel": cfg.panel,
                "dash_mode": cfg.dash_mode,
                "point_order": order,
                "x_value": float(x_data[0]),
                "y_value": float(y_data[0]),
                "pixel_x": float(x + cfg.axis.left),
                "pixel_y": float(y + cfg.axis.top),
                "local_x": x,
                "local_y": y,
                "candidate_count": len(values),
                "source": "endpoint_visible_dash" if cfg.dash_mode == "visible_only" else "endpoint_trace",
            }
        )
    return rows


def profile_diagnostics(mask: np.ndarray, profile: list[dict[str, object]]) -> dict[str, float | int | str]:
    by_y = {int(row["local_y"]): float(row["local_x"]) for row in profile}
    mask_y, mask_x = np.where(mask)
    residuals = [abs(float(x) - by_y[int(y)]) for y, x in zip(mask_y, mask_x) if int(y) in by_y]
    y_values = np.array(sorted(by_y), dtype=float)
    x_values = np.array([by_y[int(y)] for y in y_values], dtype=float)
    gaps = np.diff(y_values) if len(y_values) > 1 else np.array([], dtype=float)
    dx = np.abs(np.diff(x_values)) if len(x_values) > 1 else np.array([], dtype=float)
    ddx = np.abs(np.diff(x_values, n=2)) if len(x_values) > 2 else np.array([], dtype=float)
    return {
        "mask_pixels": int(mask.sum()),
        "mask_rows_covered": len(set(int(y) for y in mask_y) & set(by_y)),
        "mask_residual_mean_px": float(np.mean(residuals)) if residuals else float("nan"),
        "mask_residual_p95_px": float(np.percentile(residuals, 95)) if residuals else float("nan"),
        "mask_residual_max_px": float(np.max(residuals)) if residuals else float("nan"),
        "max_y_gap_rows": int(np.max(gaps)) if len(gaps) else 0,
        "p95_row_dx_px": float(np.percentile(dx, 95)) if len(dx) else 0.0,
        "p95_second_diff_px": float(np.percentile(ddx, 95)) if len(ddx) else 0.0,
        "duplicate_y_rows": len(profile) - len({round(float(row["y_value"]), 6) for row in profile}),
    }


def extract_series(rgb: np.ndarray, cfg: SeriesConfig) -> SeriesResult:
    crop = crop_series(rgb, cfg.axis)
    mask = build_mask(crop, cfg)
    comps = extract_components(mask, cfg)
    points, links = endpoint_polyline(comps, cfg)
    profile = profile_from_polyline(points, cfg)
    diagnostics = profile_diagnostics(mask, profile)
    return SeriesResult(cfg, mask, comps, links, profile, diagnostics)


def write_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def format_row(row: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        out[key] = f"{value:.6f}" if isinstance(value, float) else value
    return out


def component_rows(result: SeriesResult) -> list[dict[str, object]]:
    return [
        {
            "series_key": result.config.key,
            "series_name": result.config.name,
            "label": comp.label,
            "area": comp.area,
            "bbox_x": comp.bbox_x + int(round(result.config.axis.left)),
            "bbox_y": comp.bbox_y + int(round(result.config.axis.top)),
            "bbox_w": comp.bbox_w,
            "bbox_h": comp.bbox_h,
            "center_x": comp.center_x + result.config.axis.left,
            "center_y": comp.center_y + result.config.axis.top,
            "top_x": comp.top_x + result.config.axis.left,
            "top_y": comp.top_y + result.config.axis.top,
            "bottom_x": comp.bottom_x + result.config.axis.left,
            "bottom_y": comp.bottom_y + result.config.axis.top,
            "length_px": comp.length_px,
            "dx_dy": comp.dx_dy,
        }
        for comp in result.components
    ]


def write_series_artifacts(rgb: np.ndarray, result: SeriesResult, out_dir: Path) -> dict[str, Path]:
    cfg = result.config
    series_dir = out_dir / cfg.key
    series_dir.mkdir(parents=True, exist_ok=True)
    profile_path = series_dir / f"{cfg.key}_endpoint_profile.csv"
    components_path = series_dir / f"{cfg.key}_endpoint_components.csv"
    links_path = series_dir / f"{cfg.key}_endpoint_links.csv"
    report_path = series_dir / f"{cfg.key}_endpoint_report.txt"
    overlay_path = series_dir / f"{cfg.key}_endpoint_overlay.png"
    local_overlay_path = series_dir / f"{cfg.key}_endpoint_local_overlay.png"
    redraw_path = series_dir / f"{cfg.key}_endpoint_redraw.png"
    xlsx_path = series_dir / f"{cfg.key}_endpoint_profile.xlsx"

    point_headers = ["series_key", "series_name", "panel", "dash_mode", "point_order", "x_value", "y_value", "pixel_x", "pixel_y", "local_x", "local_y", "candidate_count", "source"]
    write_csv(profile_path, [format_row(row) for row in result.profile], point_headers)
    write_csv(components_path, [format_row(row) for row in component_rows(result)], ["series_key", "series_name", "label", "area", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "center_x", "center_y", "top_x", "top_y", "bottom_x", "bottom_y", "length_px", "dx_dy"])
    write_csv(links_path, [format_row(row) for row in result.links], ["from_label", "to_label", "from_bottom_x", "from_bottom_y", "to_top_x", "to_top_y", "gap_x_px", "gap_y_px", "distance_px", "status"])

    draw_series_overlays(rgb, result, overlay_path, local_overlay_path)
    draw_series_redraw(result, redraw_path)
    write_series_excel(result, xlsx_path)
    write_series_report(result, report_path, profile_path, components_path, links_path, overlay_path, local_overlay_path, redraw_path, xlsx_path)
    return {
        "profile": profile_path,
        "components": components_path,
        "links": links_path,
        "report": report_path,
        "overlay": overlay_path,
        "local_overlay": local_overlay_path,
        "redraw": redraw_path,
        "xlsx": xlsx_path,
    }


def draw_series_overlays(rgb: np.ndarray, result: SeriesResult, overlay_path: Path, local_overlay_path: Path) -> None:
    cfg = result.config
    crop = crop_series(rgb, cfg.axis)
    local = Image.fromarray(crop).convert("RGBA")
    local = Image.alpha_composite(local, Image.new("RGBA", local.size, (255, 255, 255, 100)))
    draw = ImageDraw.Draw(local)
    yy, xx = np.where(result.mask)
    for x, y in zip(xx.tolist(), yy.tolist()):
        draw.point((x, y), fill=(255, 0, 0, 160))
    if cfg.dash_mode == "continuous":
        pts = [(row["local_x"], row["local_y"]) for row in result.profile]
        if len(pts) > 1:
            draw.line(pts, fill="#00aa00ff", width=2)
    for comp in result.components:
        draw.line([(comp.top_x, comp.top_y), (comp.bottom_x, comp.bottom_y)], fill="#0080ffff", width=1)
        draw.ellipse([comp.top_x - 1.5, comp.top_y - 1.5, comp.top_x + 1.5, comp.top_y + 1.5], fill="#00a0ffff")
        draw.ellipse([comp.bottom_x - 1.5, comp.bottom_y - 1.5, comp.bottom_x + 1.5, comp.bottom_y + 1.5], fill="#0040ffff")
    local.convert("RGB").save(local_overlay_path)

    full = Image.fromarray(rgb).convert("RGBA")
    full = Image.alpha_composite(full, Image.new("RGBA", full.size, (255, 255, 255, 95)))
    draw = ImageDraw.Draw(full)
    pts = [(float(row["pixel_x"]), float(row["pixel_y"])) for row in result.profile]
    if cfg.dash_mode == "continuous" and len(pts) > 1:
        draw.line(pts, fill=cfg.line_color + "ff", width=3, joint="curve")
    else:
        for x, y in pts:
            draw.ellipse([x - 2.0, y - 2.0, x + 2.0, y + 2.0], fill=cfg.line_color + "ff")
    for comp in result.components:
        draw.line(
            [(comp.top_x + cfg.axis.left, comp.top_y + cfg.axis.top), (comp.bottom_x + cfg.axis.left, comp.bottom_y + cfg.axis.top)],
            fill="#00a0ffff",
            width=1,
        )
    full.convert("RGB").save(overlay_path)


def draw_series_redraw(result: SeriesResult, out_path: Path) -> None:
    cfg = result.config
    fig, ax = plt.subplots(figsize=(3.2, 7.0), dpi=180)
    x = [float(row["x_value"]) for row in result.profile]
    y = [float(row["y_value"]) for row in result.profile]
    if x:
        linestyle = "None" if cfg.dash_mode == "visible_only" else "-"
        marker = "o" if cfg.dash_mode == "visible_only" else None
        ax.plot(x, y, color=cfg.line_color, linestyle=linestyle, marker=marker, markersize=2.4, linewidth=1.5)
    ax.set_xlim(cfg.axis.x_min, cfg.axis.x_max)
    if cfg.axis.reverse_y:
        ax.set_ylim(cfg.axis.y_max, cfg.axis.y_min)
    else:
        ax.set_ylim(cfg.axis.y_min, cfg.axis.y_max)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(cfg.name)
    ax.set_ylabel("y")
    ax.grid(True, color="#999999", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def write_series_excel(result: SeriesResult, out_path: Path) -> None:
    workbook = xlsxwriter.Workbook(str(out_path))
    ws = workbook.add_worksheet("profile")
    headers = ["series_key", "series_name", "panel", "dash_mode", "point_order", "x_value", "y_value", "pixel_x", "pixel_y", "candidate_count", "source"]
    for col, header in enumerate(headers):
        ws.write(0, col, header)
    for row_idx, row in enumerate(result.profile, start=1):
        for col, header in enumerate(headers):
            value = row[header]
            if isinstance(value, (int, float)):
                ws.write_number(row_idx, col, float(value))
            else:
                ws.write(row_idx, col, value)
    chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    if result.profile:
        chart.add_series(
            {
                "name": result.config.name,
                "categories": ["profile", 1, 5, len(result.profile), 5],
                "values": ["profile", 1, 6, len(result.profile), 6],
                "line": {"color": result.config.line_color, "width": 1.5} if result.config.dash_mode == "continuous" else {"none": True},
                "marker": {"type": "none"} if result.config.dash_mode == "continuous" else {"type": "circle", "size": 4, "border": {"color": result.config.line_color}, "fill": {"color": result.config.line_color}},
            }
        )
    chart.set_x_axis({"name": result.config.name, "min": result.config.axis.x_min, "max": result.config.axis.x_max})
    y_axis: dict[str, Any] = {"name": "y", "min": result.config.axis.y_min, "max": result.config.axis.y_max}
    if result.config.axis.reverse_y:
        y_axis["reverse"] = True
    chart.set_y_axis(y_axis)
    chart.set_size({"width": 420, "height": 720})
    ws.insert_chart("M2", chart)
    workbook.close()


def write_series_report(result: SeriesResult, report_path: Path, profile_path: Path, components_path: Path, links_path: Path, overlay_path: Path, local_overlay_path: Path, redraw_path: Path, xlsx_path: Path) -> None:
    link_status: dict[str, int] = {}
    for row in result.links:
        status = str(row["status"])
        link_status[status] = link_status.get(status, 0) + 1
    d = result.diagnostics
    lines = [
        f"Endpoint dashed profile: {result.config.key}",
        f"Name: {result.config.name}",
        f"Panel: {result.config.panel}",
        f"Dash mode: {result.config.dash_mode}",
        f"Mask pixels: {d['mask_pixels']}",
        f"Dash components: {len(result.components)}",
        f"Endpoint links: {len(result.links)}; statuses={link_status}",
        f"Profile rows: {len(result.profile)}; duplicate_y_rows={d['duplicate_y_rows']}",
        f"Mask residual mean/p95/max: {float(d['mask_residual_mean_px']):.3f}/{float(d['mask_residual_p95_px']):.3f}/{float(d['mask_residual_max_px']):.3f} px",
        f"Profile diagnostics: mask_rows_covered={d['mask_rows_covered']}; max_y_gap={d['max_y_gap_rows']}; p95_row_dx={float(d['p95_row_dx_px']):.3f} px; p95_second_diff={float(d['p95_second_diff_px']):.3f} px",
        "Method: each dash component is represented by PCA top/bottom endpoints.",
        "continuous connects consecutive dash endpoints; visible_only does not connect blank gaps.",
        "",
        f"Profile CSV: {profile_path}",
        f"Components CSV: {components_path}",
        f"Links CSV: {links_path}",
        f"Overlay: {overlay_path}",
        f"Local overlay: {local_overlay_path}",
        f"Redraw: {redraw_path}",
        f"Excel: {xlsx_path}",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_combined_outputs(rgb: np.ndarray, prefix: str, results: list[SeriesResult], out_dir: Path) -> dict[str, Path]:
    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for result in results:
        long_rows.extend(format_row(row) for row in result.profile)
        d = result.diagnostics
        status_values = sorted({str(row["status"]) for row in result.links})
        summary_rows.append(
            {
                "series_key": result.config.key,
                "series_name": result.config.name,
                "panel": result.config.panel,
                "dash_mode": result.config.dash_mode,
                "components": len(result.components),
                "links": len(result.links),
                "link_statuses": ";".join(status_values),
                "profile_rows": len(result.profile),
                "duplicate_y_rows": d["duplicate_y_rows"],
                "mask_residual_mean_px": d["mask_residual_mean_px"],
                "mask_residual_p95_px": d["mask_residual_p95_px"],
                "mask_residual_max_px": d["mask_residual_max_px"],
                "max_y_gap_rows": d["max_y_gap_rows"],
                "p95_row_dx_px": d["p95_row_dx_px"],
                "p95_second_diff_px": d["p95_second_diff_px"],
            }
        )

    long_csv = out_dir / f"{prefix}_endpoint_profiles_long.csv"
    summary_csv = out_dir / f"{prefix}_endpoint_summary.csv"
    wide_csv = out_dir / f"{prefix}_endpoint_profiles_wide.csv"
    overlay = out_dir / f"{prefix}_endpoint_overlay.png"
    xlsx = out_dir / f"{prefix}_endpoint_profiles.xlsx"
    report = out_dir / f"{prefix}_endpoint_report.txt"

    point_headers = ["series_key", "series_name", "panel", "dash_mode", "point_order", "x_value", "y_value", "pixel_x", "pixel_y", "local_x", "local_y", "candidate_count", "source"]
    write_csv(long_csv, long_rows, point_headers)
    write_csv(summary_csv, [format_row(row) for row in summary_rows], ["series_key", "series_name", "panel", "dash_mode", "components", "links", "link_statuses", "profile_rows", "duplicate_y_rows", "mask_residual_mean_px", "mask_residual_p95_px", "mask_residual_max_px", "max_y_gap_rows", "p95_row_dx_px", "p95_second_diff_px"])
    write_wide_csv(results, wide_csv)
    draw_combined_overlay(rgb, results, overlay)
    write_combined_excel(results, summary_rows, long_csv, wide_csv, summary_csv, overlay, xlsx)
    write_combined_report(results, report, long_csv, wide_csv, summary_csv, overlay, xlsx)
    return {"long_csv": long_csv, "wide_csv": wide_csv, "summary_csv": summary_csv, "overlay": overlay, "xlsx": xlsx, "report": report}


def write_wide_csv(results: list[SeriesResult], path: Path) -> None:
    by_y: dict[float, dict[str, float]] = {}
    for result in results:
        for row in result.profile:
            y_key = round(float(row["y_value"]), 6)
            by_y.setdefault(y_key, {})[result.config.key] = float(row["x_value"])
    keys = [result.config.key for result in results]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["y_value", *keys])
        writer.writeheader()
        for y_value in sorted(by_y):
            row: dict[str, object] = {"y_value": f"{y_value:.6f}"}
            for key in keys:
                row[key] = f"{by_y[y_value][key]:.6f}" if key in by_y[y_value] else ""
            writer.writerow(row)


def draw_combined_overlay(rgb: np.ndarray, results: list[SeriesResult], out_path: Path) -> None:
    image = Image.fromarray(rgb).convert("RGBA")
    canvas = Image.alpha_composite(image, Image.new("RGBA", image.size, (255, 255, 255, 95)))
    draw = ImageDraw.Draw(canvas)
    for result in results:
        pts = [(float(row["pixel_x"]), float(row["pixel_y"])) for row in result.profile]
        pts.sort(key=lambda item: item[1])
        if result.config.dash_mode == "visible_only":
            for x, y in pts:
                draw.ellipse([x - 1.9, y - 1.9, x + 1.9, y + 1.9], fill=result.config.line_color + "ff")
        elif len(pts) > 1:
            draw.line(pts, fill=result.config.line_color + "ff", width=3, joint="curve")
    canvas.convert("RGB").save(out_path)


def write_combined_excel(results: list[SeriesResult], summaries: list[dict[str, object]], long_csv: Path, wide_csv: Path, summary_csv: Path, overlay_path: Path, out_path: Path) -> None:
    workbook = xlsxwriter.Workbook(str(out_path))
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    ws = workbook.add_worksheet("long_data")
    headers = ["series_key", "series_name", "panel", "dash_mode", "point_order", "x_value", "y_value", "pixel_x", "pixel_y", "candidate_count", "source"]
    for col, header in enumerate(headers):
        ws.write(0, col, header, header_fmt)
    row_idx = 1
    ranges: dict[str, tuple[int, int, SeriesConfig]] = {}
    for result in results:
        start = row_idx
        for row in result.profile:
            for col, header in enumerate(headers):
                value = row[header]
                if isinstance(value, (int, float)):
                    ws.write_number(row_idx, col, float(value))
                else:
                    ws.write(row_idx, col, value)
            row_idx += 1
        ranges[result.config.key] = (start, row_idx - 1, result.config)
    ws.freeze_panes(1, 0)

    summary_ws = workbook.add_worksheet("summary")
    summary_headers = ["series_key", "series_name", "panel", "dash_mode", "components", "links", "link_statuses", "profile_rows", "duplicate_y_rows", "mask_residual_p95_px", "max_y_gap_rows"]
    for col, header in enumerate(summary_headers):
        summary_ws.write(0, col, header, header_fmt)
    for row_idx, row in enumerate(summaries, start=1):
        for col, header in enumerate(summary_headers):
            value = row[header]
            if isinstance(value, (int, float)):
                summary_ws.write_number(row_idx, col, float(value))
            else:
                summary_ws.write(row_idx, col, value)

    chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    for _key, (start, end, cfg) in ranges.items():
        if end < start:
            continue
        chart.add_series(
            {
                "name": cfg.name,
                "categories": ["long_data", start, 5, end, 5],
                "values": ["long_data", start, 6, end, 6],
                "line": {"color": cfg.line_color, "width": 1.5} if cfg.dash_mode == "continuous" else {"none": True},
                "marker": {"type": "none"} if cfg.dash_mode == "continuous" else {"type": "circle", "size": 4, "border": {"color": cfg.line_color}, "fill": {"color": cfg.line_color}},
            }
        )
    chart.set_legend({"position": "right"})
    chart.set_size({"width": 760, "height": 520})
    ws.insert_chart("M2", chart)

    overlay_ws = workbook.add_worksheet("overlay")
    overlay_ws.write("A1", "Endpoint dashed overlay", header_fmt)
    overlay_ws.insert_image("A3", str(overlay_path), {"x_scale": 0.65, "y_scale": 0.65})
    overlay_ws.write("A40", f"Long CSV: {long_csv}")
    overlay_ws.write("A41", f"Wide CSV: {wide_csv}")
    overlay_ws.write("A42", f"Summary CSV: {summary_csv}")
    workbook.close()


def write_combined_report(results: list[SeriesResult], out_path: Path, long_csv: Path, wide_csv: Path, summary_csv: Path, overlay: Path, xlsx: Path) -> None:
    lines = [
        "Endpoint dashed profile extraction",
        "Model: each visible dash is reduced to top/bottom PCA endpoints.",
        "continuous mode connects adjacent dash endpoints; visible_only mode keeps only visible dash rows.",
        "",
        "Series summary:",
    ]
    for result in results:
        d = result.diagnostics
        lines.append(
            f"- {result.config.key}: mode={result.config.dash_mode}; components={len(result.components)}; "
            f"profile_rows={len(result.profile)}; duplicate_y_rows={d['duplicate_y_rows']}; "
            f"residual_p95={float(d['mask_residual_p95_px']):.3f}px; max_y_gap={d['max_y_gap_rows']}"
        )
    lines.extend(["", f"Long CSV: {long_csv}", f"Wide CSV: {wide_csv}", f"Summary CSV: {summary_csv}", f"Overlay: {overlay}", f"Excel: {xlsx}"])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Digitize dashed depth-profile curves by connecting dash endpoints from a JSON config.")
    parser.add_argument("--input", required=True, help="Input chart image.")
    parser.add_argument("--config", required=True, help="JSON config with per-series axes and dashed tracing settings.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix, configs = read_config(Path(args.config))
    rgb = np.array(Image.open(input_path).convert("RGB"))
    results = [extract_series(rgb, cfg) for cfg in configs]
    for result in results:
        paths = write_series_artifacts(rgb, result, out_dir)
        print(paths["report"])
    combined = write_combined_outputs(rgb, prefix, results, out_dir)
    for path in combined.values():
        print(path)


if __name__ == "__main__":
    main()
