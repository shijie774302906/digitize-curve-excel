from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class Calibration:
    left: float
    right: float
    top: float
    bottom: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    reverse_y: bool = False

    def pixel_to_data(self, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.x_min + (px - self.left) / (self.right - self.left) * (self.x_max - self.x_min)
        if self.reverse_y:
            y = self.y_min + (py - self.top) / (self.bottom - self.top) * (self.y_max - self.y_min)
        else:
            y = self.y_min + (self.bottom - py) / (self.bottom - self.top) * (self.y_max - self.y_min)
        return x, y

    def data_to_pixel(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = self.left + (x - self.x_min) / (self.x_max - self.x_min) * (self.right - self.left)
        if self.reverse_y:
            py = self.top + (y - self.y_min) / (self.y_max - self.y_min) * (self.bottom - self.top)
        else:
            py = self.bottom - (y - self.y_min) / (self.y_max - self.y_min) * (self.bottom - self.top)
        return px, py


@dataclass(frozen=True)
class PlotConfig:
    output_prefix: str
    calibration: Calibration
    x_label: str
    y_label: str
    x_major: float | None
    y_major: float | None
    exclude_rects: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class SeriesConfig:
    key: str
    name: str
    color: str
    style: str
    centers: tuple[tuple[int, int, int], ...]
    max_dist: float
    roi: tuple[float, float, float, float]
    min_area: int = 8
    component_mode: str = "all"
    close_iterations: int = 0
    guide: tuple[tuple[float, float], ...] = ()
    guide_tol_y: float = 2.0
    color_space: str = "rgb"
    min_chroma: float = 18.0
    path_order: str = "x"
    single_value_axis: str = "none"
    point_guide_tol_y: float | None = None
    interpolate_gap_px: float = 0.0


@dataclass
class SeriesResult:
    config: SeriesConfig
    points: list[dict[str, float | int | str]] = field(default_factory=list)
    components: list[dict[str, float | int | str]] = field(default_factory=list)
    rejected_components: int = 0
    rejected_by_guide: int = 0
    mask: np.ndarray | None = None


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


def read_config(path: Path) -> tuple[PlotConfig, list[SeriesConfig]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    axes = raw["axes"]
    calibration = Calibration(
        left=float(axes["left"]),
        right=float(axes["right"]),
        top=float(axes["top"]),
        bottom=float(axes["bottom"]),
        x_min=float(axes["x_min"]),
        x_max=float(axes["x_max"]),
        y_min=float(axes["y_min"]),
        y_max=float(axes["y_max"]),
        reverse_y=bool(axes.get("reverse_y", raw.get("reverse_y", False))),
    )

    labels = raw.get("labels", {})
    ticks = raw.get("ticks", {})
    exclude_rects: list[tuple[int, int, int, int]] = []
    for item in raw.get("exclude_regions", []):
        rect = item.get("rect", item)
        if len(rect) != 4:
            raise ValueError(f"Invalid exclude region: {item!r}")
        exclude_rects.append(tuple(int(round(float(v))) for v in rect))

    plot = PlotConfig(
        output_prefix=slugify(str(raw.get("output_prefix", path.stem))),
        calibration=calibration,
        x_label=str(labels.get("x", "x")),
        y_label=str(labels.get("y", "y")),
        x_major=float(ticks["x_major"]) if "x_major" in ticks else None,
        y_major=float(ticks["y_major"]) if "y_major" in ticks else None,
        exclude_rects=tuple(exclude_rects),
    )

    series: list[SeriesConfig] = []
    seen_keys: set[str] = set()
    for item in raw["series"]:
        name = str(item["name"])
        key = slugify(str(item.get("key", name)))
        base_key = key
        suffix = 2
        while key in seen_keys:
            key = f"{base_key}_{suffix}"
            suffix += 1
        seen_keys.add(key)

        color = str(item.get("color", item.get("color_hint", "#000000")))
        centers_raw = item.get("color_centers", item.get("centers"))
        if centers_raw:
            centers = tuple(tuple(int(round(float(channel))) for channel in center) for center in centers_raw)
        else:
            centers = (hex_to_rgb(color),)
        if any(len(center) != 3 for center in centers):
            raise ValueError(f"Invalid RGB center for series {name!r}")

        style = str(item.get("style", "solid")).lower()
        component_mode = str(item.get("component_mode", infer_component_mode(style))).lower()
        guide_raw = item.get("guide_points", item.get("guide", []))
        guide = tuple((float(point[0]), float(point[1])) for point in guide_raw)
        roi_raw = item.get("roi")
        if roi_raw is None:
            raise ValueError(f"Series {name!r} must define roi=[x_min,x_max,y_min,y_max]")

        series.append(
            SeriesConfig(
                key=key,
                name=name,
                color=color,
                style=style,
                centers=centers,
                max_dist=float(item.get("max_dist", 60.0)),
                roi=tuple(float(v) for v in roi_raw),
                min_area=int(item.get("min_area", 8)),
                component_mode=component_mode,
                close_iterations=int(item.get("close_iterations", 0)),
                guide=guide,
                guide_tol_y=float(item.get("guide_tol_y", 2.0)),
                color_space=str(item.get("color_space", raw.get("color_space", "rgb"))).lower(),
                min_chroma=float(item.get("min_chroma", raw.get("min_chroma", 18.0))),
                path_order=str(item.get("path_order", raw.get("path_order", "x"))).lower(),
                single_value_axis=str(item.get("single_value_axis", raw.get("single_value_axis", "none"))).lower(),
                point_guide_tol_y=(
                    float(item["point_guide_tol_y"])
                    if "point_guide_tol_y" in item and item.get("point_guide_tol_y") is not None
                    else None
                ),
                interpolate_gap_px=float(item.get("interpolate_gap_px", 0.0)),
            )
        )
    return plot, series


def infer_component_mode(style: str) -> str:
    if style == "dotted":
        return "dot"
    if style == "dashed":
        return "short"
    if style == "box":
        return "box"
    return "long"


def crop_plot(rgb: np.ndarray, calib: Calibration) -> np.ndarray:
    return rgb[int(calib.top) : int(calib.bottom) + 1, int(calib.left) : int(calib.right) + 1]


def base_allowed_mask(shape: tuple[int, int], plot: PlotConfig) -> np.ndarray:
    calib = plot.calibration
    allowed = np.ones(shape, dtype=bool)
    for x0, y0, x1, y1 in plot.exclude_rects:
        x0 = max(int(round(x0 - calib.left)), 0)
        x1 = min(int(round(x1 - calib.left)), shape[1] - 1)
        y0 = max(int(round(y0 - calib.top)), 0)
        y1 = min(int(round(y1 - calib.top)), shape[0] - 1)
        if x1 >= x0 and y1 >= y0:
            allowed[y0 : y1 + 1, x0 : x1 + 1] = False
    return allowed


def roi_mask(shape: tuple[int, int], plot: PlotConfig, roi: tuple[float, float, float, float]) -> np.ndarray:
    calib = plot.calibration
    x0, x1, y0, y1 = roi
    px0, py1 = calib.data_to_pixel(np.array([x0]), np.array([y0]))
    px1, py0 = calib.data_to_pixel(np.array([x1]), np.array([y1]))
    lx0 = max(0, int(np.floor(min(px0[0], px1[0]) - calib.left)))
    lx1 = min(shape[1] - 1, int(np.ceil(max(px0[0], px1[0]) - calib.left)))
    ly0 = max(0, int(np.floor(min(py0[0], py1[0]) - calib.top)))
    ly1 = min(shape[0] - 1, int(np.ceil(max(py0[0], py1[0]) - calib.top)))
    out = np.zeros(shape, dtype=bool)
    out[ly0 : ly1 + 1, lx0 : lx1 + 1] = True
    return out


def color_distance_mask(crop: np.ndarray, plot: PlotConfig, cfg: SeriesConfig) -> np.ndarray:
    arr_rgb = crop.astype(np.float32)
    if cfg.color_space == "lab":
        arr = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
        centers_rgb = np.array(cfg.centers, dtype=np.uint8).reshape(-1, 1, 3)
        centers = cv2.cvtColor(centers_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    elif cfg.color_space == "rgb":
        arr = arr_rgb
        centers = np.array(cfg.centers, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported color_space={cfg.color_space!r}; use rgb or lab")

    diff = arr[:, :, None, :] - centers[None, None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=3)).min(axis=2)
    mx = arr_rgb.max(axis=2)
    mn = arr_rgb.min(axis=2)
    colored = (mx - mn) >= cfg.min_chroma
    mask = (dist <= cfg.max_dist) & colored
    mask &= base_allowed_mask(mask.shape, plot)
    mask &= roi_mask(mask.shape, plot, cfg.roi)
    if cfg.close_iterations:
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iterations) > 0
    return mask


def component_is_kept(area: int, width: int, height: int, cfg: SeriesConfig) -> bool:
    max_dim = max(width, height)
    min_dim = max(1, min(width, height))
    if area < cfg.min_area:
        return False
    if cfg.component_mode == "long":
        return max_dim >= 22 or area >= 60
    if cfg.component_mode == "short":
        return max_dim <= 55 and width >= 2 and height >= 2
    if cfg.component_mode == "dot":
        return 2 <= max_dim <= 20 and area <= 80
    if cfg.component_mode == "box":
        return max_dim >= 8 and min_dim >= 1
    return True


def guide_y_at_x(cfg: SeriesConfig, x_value: float) -> float | None:
    if not cfg.guide:
        return None
    guide = sorted(cfg.guide, key=lambda item: item[0])
    xs = np.array([item[0] for item in guide], dtype=float)
    ys = np.array([item[1] for item in guide], dtype=float)
    if x_value < xs.min() - 1e-9 or x_value > xs.max() + 1e-9:
        return None
    return float(np.interp(x_value, xs, ys))


def near_guide(cfg: SeriesConfig, x_values: np.ndarray, y_values: np.ndarray) -> bool:
    if not cfg.guide or cfg.component_mode == "box":
        return True
    x_center = float(np.median(x_values))
    y_center = float(np.median(y_values))
    expected = guide_y_at_x(cfg, x_center)
    if expected is None:
        return False
    return abs(y_center - expected) <= cfg.guide_tol_y


def ordered_component_points(component_mask: np.ndarray, cfg: SeriesConfig) -> np.ndarray:
    if cfg.component_mode == "dot":
        ys, xs = np.where(component_mask)
        return np.array([[float(np.median(xs)), float(np.median(ys))]], dtype=float)

    if cfg.component_mode == "box":
        contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
            return contour

    skel = skeletonize(component_mask).astype(bool)
    ys, xs = np.where(skel)
    if len(xs) == 0:
        ys, xs = np.where(component_mask)
    points = np.column_stack([xs.astype(float), ys.astype(float)])
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)

    if cfg.path_order == "y":
        order = np.lexsort((points[:, 0], points[:, 1]))
    else:
        order = np.lexsort((points[:, 1], points[:, 0]))
    return points[order]


def extract_manual_box(plot: PlotConfig, cfg: SeriesConfig) -> SeriesResult:
    if not cfg.guide:
        raise ValueError(f"Series {cfg.name!r} uses component_mode=box and must define guide_points")
    calib = plot.calibration
    result = SeriesResult(config=cfg, mask=None)
    px, py = calib.data_to_pixel(
        np.array([point[0] for point in cfg.guide], dtype=float),
        np.array([point[1] for point in cfg.guide], dtype=float),
    )
    result.components.append(
        {
            "series_key": cfg.key,
            "series_name": cfg.name,
            "component_label": 1,
            "area_px": 0,
            "bbox_x": int(np.min(px)),
            "bbox_y": int(np.min(py)),
            "bbox_width": int(np.max(px) - np.min(px)),
            "bbox_height": int(np.max(py) - np.min(py)),
            "point_count": len(cfg.guide),
            "x_min": min(point[0] for point in cfg.guide),
            "x_max": max(point[0] for point in cfg.guide),
            "y_min": min(point[1] for point in cfg.guide),
            "y_max": max(point[1] for point in cfg.guide),
            "component_order": 1,
        }
    )
    for order, ((x_value, y_value), pixel_x, pixel_y) in enumerate(zip(cfg.guide, px, py), start=1):
        result.points.append(
            {
                "series_key": cfg.key,
                "series_name": cfg.name,
                "line_style": cfg.style,
                "component_label": 1,
                "component_order": 1,
                "point_order": order,
                "x_value": float(x_value),
                "y_value": float(y_value),
                "pixel_x": float(pixel_x),
                "pixel_y": float(pixel_y),
            }
        )
    return result


def extract_x_profile(crop: np.ndarray, plot: PlotConfig, cfg: SeriesConfig) -> SeriesResult:
    calib = plot.calibration
    mask = color_distance_mask(crop, plot, cfg)
    result = SeriesResult(config=cfg, mask=mask)
    ys_all, xs_all = np.where(mask)
    if len(xs_all) == 0:
        return result

    order = 1
    kept_pixels: list[tuple[float, float, float, float]] = []
    guide_tol = cfg.point_guide_tol_y if cfg.point_guide_tol_y is not None else cfg.guide_tol_y
    for local_x in sorted(set(int(value) for value in xs_all)):
        local_ys = ys_all[xs_all == local_x]
        global_x = np.full(len(local_ys), calib.left + local_x, dtype=float)
        global_y = calib.top + local_ys.astype(float)
        x_values, y_values = calib.pixel_to_data(global_x, global_y)
        expected = guide_y_at_x(cfg, float(x_values[0]))
        if expected is not None:
            distances = np.abs(y_values - expected)
            allowed = distances <= guide_tol
            if not np.any(allowed):
                continue
            candidate_idx = int(np.where(allowed)[0][np.argmin(distances[allowed])])
        else:
            candidate_idx = int(np.argmin(np.abs(y_values - np.median(y_values))))
        pixel_x = float(global_x[candidate_idx])
        pixel_y = float(global_y[candidate_idx])
        x_value = float(x_values[candidate_idx])
        y_value = float(y_values[candidate_idx])
        kept_pixels.append((pixel_x, pixel_y, x_value, y_value))
        result.points.append(
            {
                "series_key": cfg.key,
                "series_name": cfg.name,
                "line_style": cfg.style,
                "component_label": 1,
                "component_order": 1,
                "point_order": order,
                "x_value": x_value,
                "y_value": y_value,
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
            }
        )
        order += 1

    if cfg.interpolate_gap_px > 0 and len(result.points) > 1:
        source = sorted(result.points, key=lambda row: float(row["pixel_x"]))
        filled: list[dict[str, float | int | str]] = []
        for current, nxt in zip(source, source[1:]):
            filled.append(current)
            x0 = float(current["pixel_x"])
            x1 = float(nxt["pixel_x"])
            gap = x1 - x0
            if gap <= 1 or gap > cfg.interpolate_gap_px:
                continue
            y0 = float(current["pixel_y"])
            y1 = float(nxt["pixel_y"])
            start = int(round(x0)) + 1
            end = int(round(x1))
            for pixel_x_int in range(start, end):
                t = (pixel_x_int - x0) / gap
                pixel_x = float(pixel_x_int)
                pixel_y = y0 + t * (y1 - y0)
                x_value, y_value = calib.pixel_to_data(np.array([pixel_x]), np.array([pixel_y]))
                filled.append(
                    {
                        "series_key": cfg.key,
                        "series_name": cfg.name,
                        "line_style": cfg.style,
                        "component_label": 1,
                        "component_order": 1,
                        "point_order": 0,
                        "x_value": float(x_value[0]),
                        "y_value": float(y_value[0]),
                        "pixel_x": pixel_x,
                        "pixel_y": float(pixel_y),
                    }
                )
        filled.append(source[-1])
        result.points = sorted(filled, key=lambda row: float(row["pixel_x"]))
        for idx, row in enumerate(result.points, start=1):
            row["point_order"] = idx

    if kept_pixels:
        arr = np.array(
            [
                (
                    float(row["pixel_x"]),
                    float(row["pixel_y"]),
                    float(row["x_value"]),
                    float(row["y_value"]),
                )
                for row in result.points
            ],
            dtype=float,
        )
        result.components.append(
            {
                "series_key": cfg.key,
                "series_name": cfg.name,
                "component_label": 1,
                "area_px": int(mask.sum()),
                "bbox_x": int(calib.left + xs_all.min()),
                "bbox_y": int(calib.top + ys_all.min()),
                "bbox_width": int(xs_all.max() - xs_all.min() + 1),
                "bbox_height": int(ys_all.max() - ys_all.min() + 1),
                "point_count": len(result.points),
                "x_min": float(np.min(arr[:, 2])),
                "x_max": float(np.max(arr[:, 2])),
                "y_min": float(np.min(arr[:, 3])),
                "y_max": float(np.max(arr[:, 3])),
                "component_order": 1,
            }
        )
    return result


def extract_series(crop: np.ndarray, plot: PlotConfig, cfg: SeriesConfig) -> SeriesResult:
    if cfg.component_mode == "box" and cfg.guide:
        return extract_manual_box(plot, cfg)
    if cfg.component_mode == "x_profile":
        return extract_x_profile(crop, plot, cfg)

    calib = plot.calibration
    mask = color_distance_mask(crop, plot, cfg)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    result = SeriesResult(config=cfg, mask=mask)
    component_paths: list[tuple[float, int, np.ndarray, dict[str, float | int | str]]] = []

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if not component_is_kept(area, width, height, cfg):
            result.rejected_components += 1
            continue
        comp = labels == label
        local_path = ordered_component_points(comp, cfg)
        if len(local_path) == 0:
            result.rejected_components += 1
            continue
        global_path = local_path + np.array([calib.left, calib.top], dtype=float)
        x_data, y_data = calib.pixel_to_data(global_path[:, 0], global_path[:, 1])
        if not near_guide(cfg, x_data, y_data):
            result.rejected_by_guide += 1
            continue
        row = {
            "series_key": cfg.key,
            "series_name": cfg.name,
            "component_label": label,
            "area_px": area,
            "bbox_x": x0 + int(calib.left),
            "bbox_y": y0 + int(calib.top),
            "bbox_width": width,
            "bbox_height": height,
            "point_count": len(local_path),
            "x_min": float(np.min(x_data)),
            "x_max": float(np.max(x_data)),
            "y_min": float(np.min(y_data)),
            "y_max": float(np.max(y_data)),
        }
        sort_key = float(np.median(y_data)) if cfg.path_order == "y" else float(np.median(x_data))
        component_paths.append((sort_key, label, np.column_stack([global_path, x_data, y_data]), row))

    component_paths.sort(key=lambda item: item[0])
    order = 1
    for component_order, (_sort_key, label, path, component_row) in enumerate(component_paths, start=1):
        component_row["component_order"] = component_order
        result.components.append(component_row)
        if cfg.style in ("dashed", "dotted"):
            point_indices = [len(path) // 2]
        else:
            step = max(1, len(path) // 80)
            point_indices = range(0, len(path), step)
        for idx in point_indices:
            px, py, x_value, y_value = path[idx]
            result.points.append(
                {
                    "series_key": cfg.key,
                    "series_name": cfg.name,
                    "line_style": cfg.style,
                    "component_label": label,
                    "component_order": component_order,
                    "point_order": order,
                    "x_value": float(x_value),
                    "y_value": float(y_value),
                    "pixel_x": float(px),
                    "pixel_y": float(py),
                }
            )
            order += 1
    return result


def format_row(row: dict[str, float | int | str]) -> dict[str, str | int]:
    out: dict[str, str | int] = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = f"{value:.6f}"
        else:
            out[key] = value
    return out


def grouped_pixel_points(result: SeriesResult) -> dict[int, list[tuple[float, float]]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in result.points:
        grouped.setdefault(int(row["component_order"]), []).append((float(row["pixel_x"]), float(row["pixel_y"])))
    return grouped


def grouped_data_points(result: SeriesResult) -> dict[int, list[tuple[float, float]]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in result.points:
        grouped.setdefault(int(row["component_order"]), []).append((float(row["x_value"]), float(row["y_value"])))
    return grouped


def guide_distance(cfg: SeriesConfig, x_value: float, y_value: float) -> float:
    expected = guide_y_at_x(cfg, x_value)
    if expected is None:
        return 0.0
    return abs(y_value - expected)


def enforce_single_value_axis(result: SeriesResult) -> None:
    axis = result.config.single_value_axis
    if axis not in {"x_to_y", "y_to_x"} or len(result.points) < 2:
        return

    groups: dict[tuple[str, int], list[dict[str, float | int | str]]] = {}
    for row in result.points:
        if axis == "x_to_y":
            bucket = int(round(float(row["pixel_x"])))
        else:
            bucket = int(round(float(row["pixel_y"])))
        groups.setdefault((str(row["series_key"]), bucket), []).append(row)

    kept: list[dict[str, float | int | str]] = []
    for (_series_key, _bucket), rows in groups.items():
        if len(rows) == 1:
            kept.append(rows[0])
            continue
        if axis == "x_to_y":
            selected = min(rows, key=lambda row: guide_distance(result.config, float(row["x_value"]), float(row["y_value"])))
        else:
            values = np.array([float(row["x_value"]) for row in rows], dtype=float)
            median_x = float(np.median(values))
            selected = min(rows, key=lambda row: abs(float(row["x_value"]) - median_x))
        kept.append(selected)

    if axis == "x_to_y":
        kept.sort(key=lambda row: (float(row["pixel_x"]), float(row["pixel_y"])))
    else:
        kept.sort(key=lambda row: (float(row["pixel_y"]), float(row["pixel_x"])))
    for order, row in enumerate(kept, start=1):
        row["component_order"] = 1
        row["point_order"] = order
    result.points = kept


def write_csvs(results: list[SeriesResult], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    csv_path = out_dir / f"{prefix}_digitized.csv"
    components_path = out_dir / f"{prefix}_components.csv"
    point_headers = [
        "series_key",
        "series_name",
        "line_style",
        "component_label",
        "component_order",
        "point_order",
        "x_value",
        "y_value",
        "pixel_x",
        "pixel_y",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=point_headers)
        writer.writeheader()
        for result in results:
            for row in result.points:
                writer.writerow(format_row(row))

    component_headers = [
        "series_key",
        "series_name",
        "component_order",
        "component_label",
        "area_px",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "point_count",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    ]
    with components_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=component_headers)
        writer.writeheader()
        for result in results:
            for row in result.components:
                writer.writerow(format_row(row))
    return csv_path, components_path


def draw_overlay(rgb: np.ndarray, results: list[SeriesResult], out_path: Path) -> None:
    overlay = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.alpha_composite(overlay, Image.new("RGBA", overlay.size, (255, 255, 255, 105)))
    draw = ImageDraw.Draw(overlay)
    for result in results:
        if result.config.style == "dotted":
            for row in result.points:
                px = float(row["pixel_x"])
                py = float(row["pixel_y"])
                draw.ellipse([px - 2.2, py - 2.2, px + 2.2, py + 2.2], fill=result.config.color + "ff")
        elif result.config.style == "solid":
            for pts in grouped_pixel_points(result).values():
                if len(pts) > 1:
                    draw.line(pts, fill=result.config.color + "ff", width=3, joint="curve")
        else:
            pts = [(float(row["pixel_x"]), float(row["pixel_y"])) for row in result.points]
            if len(pts) > 1:
                draw.line(pts, fill=result.config.color + "ff", width=3, joint="curve")
    overlay.convert("RGB").save(out_path)


def draw_redraw(results: list[SeriesResult], plot: PlotConfig, out_path: Path) -> None:
    calib = plot.calibration
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=150)
    for result in results:
        if result.config.style == "dotted":
            x = [float(row["x_value"]) for row in result.points]
            y = [float(row["y_value"]) for row in result.points]
            ax.plot(x, y, color=result.config.color, linestyle="None", marker="o", markersize=2.5, label=result.config.name)
        elif result.config.style == "solid":
            first = True
            for pts in grouped_data_points(result).values():
                if len(pts) < 2:
                    continue
                x = [point[0] for point in pts]
                y = [point[1] for point in pts]
                ax.plot(x, y, color=result.config.color, linestyle="-", linewidth=1.8, label=result.config.name if first else None)
                first = False
        else:
            x = [float(row["x_value"]) for row in result.points]
            y = [float(row["y_value"]) for row in result.points]
            if not x:
                continue
            linestyle = "-" if result.config.style == "box" else "--"
            ax.plot(x, y, color=result.config.color, linestyle=linestyle, linewidth=1.8, label=result.config.name)
    ax.set_xlim(calib.x_min, calib.x_max)
    if calib.reverse_y:
        ax.set_ylim(calib.y_max, calib.y_min)
    else:
        ax.set_ylim(calib.y_min, calib.y_max)
    ax.set_xlabel(plot.x_label)
    ax.set_ylabel(plot.y_label)
    if plot.x_major:
        ax.set_xticks(np.arange(calib.x_min, calib.x_max + plot.x_major * 0.5, plot.x_major))
    if plot.y_major:
        ax.set_yticks(np.arange(calib.y_min, calib.y_max + plot.y_major * 0.5, plot.y_major))
    ax.grid(True, color="#777777", linewidth=0.55, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def write_excel(results: list[SeriesResult], plot: PlotConfig, out_path: Path) -> None:
    workbook = xlsxwriter.Workbook(str(out_path))
    ws = workbook.add_worksheet("data")
    headers = ["series_name", "line_style", "point_order", "x_value", "y_value", "pixel_x", "pixel_y"]
    for col, header in enumerate(headers):
        ws.write(0, col, header)
    row_idx = 1
    starts: dict[str, tuple[int, int, SeriesConfig]] = {}
    for result in results:
        start = row_idx
        for row in result.points:
            ws.write(row_idx, 0, row["series_name"])
            ws.write(row_idx, 1, row["line_style"])
            ws.write_number(row_idx, 2, int(row["point_order"]))
            ws.write_number(row_idx, 3, float(row["x_value"]))
            ws.write_number(row_idx, 4, float(row["y_value"]))
            ws.write_number(row_idx, 5, float(row["pixel_x"]))
            ws.write_number(row_idx, 6, float(row["pixel_y"]))
            row_idx += 1
        starts[result.config.key] = (start, row_idx - 1, result.config)

    comp = workbook.add_worksheet("components")
    comp_headers = ["series_name", "component_order", "area_px", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "point_count"]
    for col, header in enumerate(comp_headers):
        comp.write(0, col, header)
    comp_row = 1
    for result in results:
        for row in result.components:
            comp.write(comp_row, 0, row["series_name"])
            comp.write_number(comp_row, 1, int(row["component_order"]))
            comp.write_number(comp_row, 2, int(row["area_px"]))
            comp.write_number(comp_row, 3, int(row["bbox_x"]))
            comp.write_number(comp_row, 4, int(row["bbox_y"]))
            comp.write_number(comp_row, 5, int(row["bbox_width"]))
            comp.write_number(comp_row, 6, int(row["bbox_height"]))
            comp.write_number(comp_row, 7, int(row["point_count"]))
            comp_row += 1

    chart = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
    for _key, (start, end, cfg) in starts.items():
        if end < start:
            continue
        line: dict[str, Any]
        marker: dict[str, Any]
        if cfg.style == "dotted":
            line = {"none": True}
            marker = {"type": "circle", "size": 4, "border": {"color": cfg.color}, "fill": {"color": cfg.color}}
        else:
            line = {"color": cfg.color, "width": 1.5}
            if cfg.style == "dashed":
                line["dash_type"] = "dash"
            marker = {"type": "none"}
        chart.add_series(
            {
                "name": cfg.name,
                "categories": ["data", start, 3, end, 3],
                "values": ["data", start, 4, end, 4],
                "line": line,
                "marker": marker,
            }
        )
    calib = plot.calibration
    x_axis: dict[str, Any] = {"name": plot.x_label, "min": calib.x_min, "max": calib.x_max}
    y_axis: dict[str, Any] = {"name": plot.y_label, "min": calib.y_min, "max": calib.y_max}
    if calib.reverse_y:
        y_axis["reverse"] = True
    if plot.x_major:
        x_axis["major_unit"] = plot.x_major
    if plot.y_major:
        y_axis["major_unit"] = plot.y_major
    chart.set_x_axis(x_axis)
    chart.set_y_axis(y_axis)
    chart.set_legend({"position": "right"})
    chart.set_size({"width": 760, "height": 520})
    ws.insert_chart("I2", chart)
    workbook.close()


def write_mask_previews(rgb: np.ndarray, plot: PlotConfig, results: list[SeriesResult], out_dir: Path, prefix: str) -> list[Path]:
    calib = plot.calibration
    paths: list[Path] = []
    for result in results:
        if result.mask is None:
            continue
        preview = Image.fromarray(rgb).convert("RGBA")
        preview = Image.alpha_composite(preview, Image.new("RGBA", preview.size, (255, 255, 255, 125)))
        color = hex_to_rgb(result.config.color)
        layer = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        y0 = int(calib.top)
        x0 = int(calib.left)
        mask = result.mask.astype(bool)
        layer[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1], 0] = color[0]
        layer[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1], 1] = color[1]
        layer[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1], 2] = color[2]
        layer[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1], 3] = (mask * 190).astype(np.uint8)
        out = Image.alpha_composite(preview, Image.fromarray(layer, mode="RGBA")).convert("RGB")
        path = out_dir / f"{prefix}_mask_{result.config.key}.png"
        out.save(path)
        paths.append(path)
    return paths


def write_overlap_diagnostics(rgb: np.ndarray, plot: PlotConfig, results: list[SeriesResult], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    masks = [(result.config.key, result.mask.astype(bool)) for result in results if result.mask is not None]
    csv_path = out_dir / f"{prefix}_mask_overlap.csv"
    overlay_path = out_dir / f"{prefix}_mask_overlap.png"

    rows: list[dict[str, int | str]] = []
    if masks:
        stack = np.stack([mask for _key, mask in masks], axis=0)
        overlap = stack.sum(axis=0)
        rows.append({"series_a": "any", "series_b": "any", "overlap_px": int(np.count_nonzero(overlap > 1))})
        for idx, (key_a, mask_a) in enumerate(masks):
            for key_b, mask_b in masks[idx + 1 :]:
                rows.append({"series_a": key_a, "series_b": key_b, "overlap_px": int(np.count_nonzero(mask_a & mask_b))})
    else:
        overlap = np.zeros((int(plot.calibration.bottom - plot.calibration.top + 1), int(plot.calibration.right - plot.calibration.left + 1)), dtype=int)
        rows.append({"series_a": "any", "series_b": "any", "overlap_px": 0})

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["series_a", "series_b", "overlap_px"])
        writer.writeheader()
        writer.writerows(rows)

    preview = Image.fromarray(rgb).convert("RGBA")
    preview = Image.alpha_composite(preview, Image.new("RGBA", preview.size, (255, 255, 255, 135)))
    layer = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    calib = plot.calibration
    y0 = int(calib.top)
    x0 = int(calib.left)
    overlap_mask = overlap > 1
    layer[y0 : y0 + overlap_mask.shape[0], x0 : x0 + overlap_mask.shape[1], 0] = 255
    layer[y0 : y0 + overlap_mask.shape[0], x0 : x0 + overlap_mask.shape[1], 1] = 190
    layer[y0 : y0 + overlap_mask.shape[0], x0 : x0 + overlap_mask.shape[1], 2] = 0
    layer[y0 : y0 + overlap_mask.shape[0], x0 : x0 + overlap_mask.shape[1], 3] = (overlap_mask * 210).astype(np.uint8)
    Image.alpha_composite(preview, Image.fromarray(layer, mode="RGBA")).convert("RGB").save(overlay_path)
    return csv_path, overlay_path


def write_report(
    plot: PlotConfig,
    results: list[SeriesResult],
    report_path: Path,
    csv_path: Path,
    components_path: Path,
    overlay_path: Path,
    redraw_path: Path,
    xlsx_path: Path,
    overlap_csv: Path,
    overlap_overlay: Path,
    mask_paths: list[Path],
) -> None:
    calib = plot.calibration
    lines = [
        "Config-driven multi-series digitization",
        f"Axes pixels: left={calib.left}, right={calib.right}, top={calib.top}, bottom={calib.bottom}",
        f"Data range: x={calib.x_min}..{calib.x_max}; y={calib.y_min}..{calib.y_max}",
        f"Excluded regions: {len(plot.exclude_rects)}",
        "",
        "Series summary:",
    ]
    for result in results:
        lines.append(
            f"- {result.config.name}: style={result.config.style}; components={len(result.components)}; "
            f"points={len(result.points)}; rejected_components={result.rejected_components}; "
            f"rejected_by_guide={result.rejected_by_guide}; color_space={result.config.color_space}; "
            f"max_dist={result.config.max_dist}"
        )
    lines.extend(
        [
            "",
            "Outputs:",
            f"CSV: {csv_path}",
            f"Components: {components_path}",
            f"Overlay: {overlay_path}",
            f"Redraw: {redraw_path}",
            f"Excel: {xlsx_path}",
            f"Mask overlap CSV: {overlap_csv}",
            f"Mask overlap overlay: {overlap_overlay}",
        ]
    )
    if mask_paths:
        lines.append("Mask previews:")
        lines.extend(f"- {path}" for path in mask_paths)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(rgb: np.ndarray, plot: PlotConfig, results: list[SeriesResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = plot.output_prefix
    for result in results:
        enforce_single_value_axis(result)
    csv_path, components_path = write_csvs(results, out_dir, prefix)
    overlay_path = out_dir / f"{prefix}_overlay.png"
    redraw_path = out_dir / f"{prefix}_redraw.png"
    xlsx_path = out_dir / f"{prefix}.xlsx"
    report_path = out_dir / f"{prefix}_report.txt"

    draw_overlay(rgb, results, overlay_path)
    draw_redraw(results, plot, redraw_path)
    write_excel(results, plot, xlsx_path)
    mask_paths = write_mask_previews(rgb, plot, results, out_dir, prefix)
    overlap_csv, overlap_overlay = write_overlap_diagnostics(rgb, plot, results, out_dir, prefix)
    write_report(plot, results, report_path, csv_path, components_path, overlay_path, redraw_path, xlsx_path, overlap_csv, overlap_overlay, mask_paths)

    for path in [csv_path, components_path, overlay_path, redraw_path, xlsx_path, report_path, overlap_csv, overlap_overlay]:
        print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Digitize multiple colored plot series from a JSON configuration.")
    parser.add_argument("--input", required=True, help="Input chart image path.")
    parser.add_argument("--config", required=True, help="JSON config with axes, exclusions, and series definitions.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    plot, series = read_config(config_path)
    rgb = np.array(Image.open(input_path).convert("RGB"))
    crop = crop_plot(rgb, plot.calibration)
    results = [extract_series(crop, plot, cfg) for cfg in series]
    write_outputs(rgb, plot, results, out_dir)


if __name__ == "__main__":
    main()
