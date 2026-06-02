from __future__ import annotations

import argparse
import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import xlsxwriter
    from matplotlib import font_manager
    from PIL import Image, ImageDraw
    from skimage.morphology import skeletonize
except ImportError as exc:  # pragma: no cover - helps users diagnose environment issues.
    raise SystemExit(
        "Missing dependency. Install or use a Python environment with: "
        "opencv-python pillow numpy matplotlib xlsxwriter scikit-image. "
        f"Original error: {exc}"
    )


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
    reverse_y: bool

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
            py = self.top + (self.y_max - y) / (self.y_max - self.y_min) * (self.bottom - self.top)
        return px, py

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        return int(round(self.left)), int(round(self.top)), int(round(self.right)), int(round(self.bottom))


@dataclass
class CurveSegment:
    segment_id: int
    pixel_path: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_min: np.ndarray | None = None
    x_max: np.ndarray | None = None
    selection_rule: list[str] | None = None


def parse_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def guide_y_at_x(args: argparse.Namespace, x_value: float) -> float | None:
    guide = getattr(args, "guide_point", None)
    if not guide:
        return None
    ordered = sorted((float(point[0]), float(point[1])) for point in guide)
    xs = np.array([point[0] for point in ordered], dtype=float)
    ys = np.array([point[1] for point in ordered], dtype=float)
    if x_value < xs.min() - 1e-9 or x_value > xs.max() + 1e-9:
        return None
    return float(np.interp(x_value, xs, ys))


def cluster_indices(indices: np.ndarray) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    clusters: list[list[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        idx = int(idx)
        if idx == clusters[-1][-1] + 1:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return [np.array(cluster, dtype=int) for cluster in clusters]


def weighted_center(cluster: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(cluster.astype(float), weights=weights[cluster].astype(float)))


def detect_axes(rgb: np.ndarray, args: argparse.Namespace) -> Calibration:
    if args.axes:
        left, right, top, bottom = [float(v) for v in args.axes]
        return Calibration(left, right, top, bottom, args.x_min, args.x_max, args.y_min, args.y_max, args.reverse_y)

    gray = rgb.mean(axis=2)
    black = gray < args.frame_threshold
    col_count = black.sum(axis=0)
    row_count = black.sum(axis=1)

    vertical = cluster_indices(np.where(col_count > rgb.shape[0] * args.frame_column_fraction)[0])
    horizontal = cluster_indices(np.where(row_count > rgb.shape[1] * args.frame_row_fraction)[0])
    if len(vertical) < 2 or len(horizontal) < 2:
        raise RuntimeError(
            "Could not auto-detect the plot frame. Re-run with --axes LEFT RIGHT TOP BOTTOM."
        )

    return Calibration(
        left=weighted_center(vertical[0], col_count),
        right=weighted_center(vertical[-1], col_count),
        top=weighted_center(horizontal[0], row_count),
        bottom=weighted_center(horizontal[-1], row_count),
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        reverse_y=args.reverse_y,
    )


def color_mask(rgb_crop: np.ndarray, preset: str) -> np.ndarray:
    r = rgb_crop[:, :, 0].astype(np.int16)
    g = rgb_crop[:, :, 1].astype(np.int16)
    b = rgb_crop[:, :, 2].astype(np.int16)
    gray = rgb_crop.mean(axis=2)

    if preset == "red":
        mask = (r > 95) & (g < 95) & (b < 95) & ((r - g) > 30) & ((r - b) > 30)
    elif preset == "blue":
        mask = (b > 80) & (r < 130) & (g < 150) & ((b - r) > 20)
    elif preset == "blue-solid":
        # Use for saturated blue data curves when cyan/teal dashed reference
        # lines or labels are present in the same plotting area.
        mask = (b > 130) & (r < 90) & (g < 110) & ((b - g) > 60) & ((b - r) > 80)
    elif preset == "green":
        mask = (g > 80) & (r < 150) & (b < 150) & ((g - r) > 15)
    elif preset == "purple":
        mask = (r > 90) & (b > 90) & (g < 150) & ((r - g) > 18) & ((b - g) > 8)
    elif preset == "dark":
        mask = gray < 95
    else:
        raise ValueError(f"Unsupported curve preset: {preset}")

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((2, 2), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_u8 > 0


def color_distance_mask(
    rgb_crop: np.ndarray,
    centers: list[list[float]] | tuple[tuple[float, float, float], ...],
    color_space: str,
    max_dist: float,
    min_chroma: float,
) -> np.ndarray:
    if not centers:
        raise ValueError("custom color extraction requires at least one --color-center R G B value")
    arr_rgb = rgb_crop.astype(np.float32)
    center_arr = np.array(centers, dtype=np.float32)
    if center_arr.ndim != 2 or center_arr.shape[1] != 3:
        raise ValueError("color centers must be RGB triples")

    if color_space == "lab":
        arr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
        centers_uint8 = np.clip(center_arr, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        centers_working = cv2.cvtColor(centers_uint8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    elif color_space == "rgb":
        arr = arr_rgb
        centers_working = center_arr
    else:
        raise ValueError(f"Unsupported color_space={color_space!r}; use rgb or lab")

    diff = arr[:, :, None, :] - centers_working[None, None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=3)).min(axis=2)
    chroma = arr_rgb.max(axis=2) - arr_rgb.min(axis=2)
    mask = (dist <= max_dist) & (chroma >= min_chroma)
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((2, 2), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_u8 > 0


def apply_roi(mask: np.ndarray, calib: Calibration, args: argparse.Namespace) -> np.ndarray:
    out = mask.copy()
    crop_left, crop_top, _crop_right, _crop_bottom = calib.crop_box
    keep = np.ones(out.shape, dtype=bool)

    def apply_global_pixel_rect(rect: list[float] | tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = [int(round(value)) for value in rect]
        lx0 = max(0, min(x0, x1) - crop_left)
        lx1 = min(out.shape[1] - 1, max(x0, x1) - crop_left)
        ly0 = max(0, min(y0, y1) - crop_top)
        ly1 = min(out.shape[0] - 1, max(y0, y1) - crop_top)
        rect_keep = np.zeros(out.shape, dtype=bool)
        if lx1 >= lx0 and ly1 >= ly0:
            rect_keep[ly0 : ly1 + 1, lx0 : lx1 + 1] = True
        keep[:] &= rect_keep

    if getattr(args, "roi", None):
        x0, x1, y0, y1 = [float(value) for value in args.roi]
        px, py = calib.data_to_pixel(
            np.array([x0, x1], dtype=float),
            np.array([y0, y1], dtype=float),
        )
        apply_global_pixel_rect((float(px[0]), float(py[0]), float(px[1]), float(py[1])))

    if getattr(args, "pixel_roi", None):
        apply_global_pixel_rect(args.pixel_roi)

    if getattr(args, "roi", None) or getattr(args, "pixel_roi", None):
        out &= keep
    return out


def curve_mask(rgb_crop: np.ndarray, calib: Calibration, args: argparse.Namespace, preset_override: str | None = None) -> np.ndarray:
    color_centers = getattr(args, "color_center", None)
    if color_centers:
        mask = color_distance_mask(
            rgb_crop,
            color_centers,
            str(getattr(args, "color_space", "rgb")).lower(),
            float(getattr(args, "max_color_dist", 60.0)),
            float(getattr(args, "min_chroma", 18.0)),
        )
    else:
        mask = color_mask(rgb_crop, preset_override or args.curve_preset)
    return apply_roi(mask, calib, args)


def apply_exclude_rects(mask: np.ndarray, calib: Calibration, args: argparse.Namespace) -> np.ndarray:
    if not getattr(args, "exclude_rect", None):
        return mask
    out = mask.copy()
    crop_left, crop_top, _crop_right, _crop_bottom = calib.crop_box
    for rect in args.exclude_rect:
        x0, y0, x1, y1 = [int(round(value)) for value in rect]
        lx0 = max(0, min(x0, x1) - crop_left)
        lx1 = min(out.shape[1] - 1, max(x0, x1) - crop_left)
        ly0 = max(0, min(y0, y1) - crop_top)
        ly1 = min(out.shape[0] - 1, max(y0, y1) - crop_top)
        if lx1 >= lx0 and ly1 >= ly0:
            out[ly0 : ly1 + 1, lx0 : lx1 + 1] = False
    return out


def apply_dash_bridge(mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if not getattr(args, "dash_bridge", False):
        return mask
    kernel_width = max(1, int(round(args.dash_bridge_x_px)))
    kernel_height = max(1, int(round(args.dash_bridge_y_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
    bridged = cv2.morphologyEx(
        mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=max(1, int(args.dash_bridge_iterations)),
    )
    return bridged > 0


def component_masks(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components: list[tuple[int, int, int, np.ndarray]] = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        top = int(stats[label, cv2.CC_STAT_TOP])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        components.append((top, left, area, labels == label))
    components.sort(key=lambda item: (item[0], item[1]))
    return [item[3] for item in components]


def skeleton_largest_component(mask: np.ndarray) -> np.ndarray:
    skel = skeletonize(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skel, 8)
    if n <= 1:
        return skel.astype(bool)
    largest = max(range(1, n), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    return labels == largest


def neighbors(pixel: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = pixel
    out: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            q = (x + dx, y + dy)
            if q in pixels:
                out.append(q)
    return out


def dijkstra(
    start: tuple[int, int],
    neighbor_cache: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int]]]:
    dist = {start: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    while heap:
        cur_dist, cur = heapq.heappop(heap)
        if cur_dist != dist[cur]:
            continue
        cx, cy = cur
        for nxt in neighbor_cache[cur]:
            nx, ny = nxt
            step = math.sqrt(2.0) if abs(nx - cx) + abs(ny - cy) == 2 else 1.0
            nd = cur_dist + step
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(heap, (nd, nxt))
    return dist, prev


def reconstruct(
    start: tuple[int, int],
    end: tuple[int, int],
    prev: dict[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int]]:
    path = [end]
    cur = end
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def longest_path(skel: np.ndarray) -> np.ndarray:
    ys, xs = np.where(skel)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    if not pixels:
        return np.empty((0, 2), dtype=float)

    neighbor_cache = {p: neighbors(p, pixels) for p in pixels}
    endpoints = [p for p, ns in neighbor_cache.items() if len(ns) == 1]
    candidates = endpoints if len(endpoints) >= 2 else list(pixels)

    best_distance = -1.0
    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_prev: dict[tuple[int, int], tuple[int, int]] | None = None
    for start in candidates:
        dist, prev = dijkstra(start, neighbor_cache)
        for end in candidates:
            if end != start and end in dist and dist[end] > best_distance:
                best_distance = dist[end]
                best_pair = (start, end)
                best_prev = prev

    if best_pair is None or best_prev is None:
        return np.array(sorted(pixels, key=lambda p: (p[1], p[0])), dtype=float)

    path = np.array(reconstruct(best_pair[0], best_pair[1], best_prev), dtype=float)
    if path[0, 1] > path[-1, 1]:
        path = path[::-1]
    return path


def trace_skeleton_paths(skel: np.ndarray) -> list[np.ndarray]:
    ys, xs = np.where(skel)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    if not pixels:
        return []

    neighbor_cache = {p: neighbors(p, pixels) for p in pixels}
    degree = {p: len(ns) for p, ns in neighbor_cache.items()}
    nodes = {p for p, d in degree.items() if d != 2}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[list[tuple[int, int]]] = []

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return tuple(sorted((a, b)))

    def walk(start: tuple[int, int], nxt: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start, nxt]
        visited_edges.add(edge_key(start, nxt))
        prev = start
        cur = nxt
        while cur not in nodes:
            candidates = [q for q in neighbor_cache[cur] if q != prev]
            if not candidates:
                break
            q = candidates[0]
            key = edge_key(cur, q)
            if key in visited_edges:
                break
            visited_edges.add(key)
            path.append(q)
            prev, cur = cur, q
        return path

    for node in sorted(nodes, key=lambda p: (p[1], p[0])):
        for nxt in sorted(neighbor_cache[node], key=lambda p: (p[1], p[0])):
            if edge_key(node, nxt) not in visited_edges:
                paths.append(walk(node, nxt))

    # Closed loops contain no graph nodes because every pixel has degree 2.
    for p in sorted(pixels, key=lambda q: (q[1], q[0])):
        open_neighbors = [q for q in neighbor_cache[p] if edge_key(p, q) not in visited_edges]
        if not open_neighbors:
            continue
        start = p
        nxt = open_neighbors[0]
        path = [start, nxt]
        visited_edges.add(edge_key(start, nxt))
        prev = start
        cur = nxt
        while True:
            candidates = [q for q in neighbor_cache[cur] if q != prev]
            if not candidates:
                break
            q = candidates[0]
            key = edge_key(cur, q)
            if key in visited_edges:
                break
            visited_edges.add(key)
            path.append(q)
            prev, cur = cur, q
            if cur == start:
                break
        paths.append(path)

    arrays = []
    for path in paths:
        arr = np.array(path, dtype=float)
        if arr[0, 1] > arr[-1, 1]:
            arr = arr[::-1]
        arrays.append(arr)
    return arrays


def continuous_edge_walk(skel: np.ndarray) -> np.ndarray:
    ys, xs = np.where(skel)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    if not pixels:
        return np.empty((0, 2), dtype=float)

    neighbor_cache = {p: sorted(neighbors(p, pixels), key=lambda q: (q[1], q[0])) for p in pixels}
    endpoints = [p for p, ns in neighbor_cache.items() if len(ns) == 1]
    start_candidates = endpoints if endpoints else list(pixels)
    start = min(start_candidates, key=lambda p: (p[1], p[0]))
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return tuple(sorted((a, b)))

    path: list[tuple[int, int]] = [start]
    stack: list[tuple[tuple[int, int], int]] = [(start, 0)]
    while stack:
        cur, next_idx = stack[-1]
        ns = neighbor_cache[cur]
        while next_idx < len(ns) and edge_key(cur, ns[next_idx]) in visited_edges:
            next_idx += 1

        if next_idx < len(ns):
            nxt = ns[next_idx]
            stack[-1] = (cur, next_idx + 1)
            visited_edges.add(edge_key(cur, nxt))
            path.append(nxt)
            stack.append((nxt, 0))
        else:
            stack.pop()
            if stack:
                parent = stack[-1][0]
                path.append(parent)

    return np.array(path, dtype=float)


def profile_path_from_mask(mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)

    path: list[tuple[float, float]] = []
    prev_x: float | None = None
    prev_y: int | None = None

    for y in rows:
        xs = np.where(mask[y])[0]
        clusters = cluster_indices(xs)
        candidates: list[tuple[float, int]] = []
        for cluster in clusters:
            center = float(cluster.mean())
            width = int(cluster[-1] - cluster[0] + 1)
            candidates.append((center, width))

        if prev_x is None:
            x = max(candidates, key=lambda item: (item[1], -item[0]))[0]
        else:
            x = min(candidates, key=lambda item: (abs(item[0] - prev_x), -item[1]))[0]

        if prev_x is not None and prev_y is not None and y - prev_y > 1:
            gap = int(y - prev_y)
            if gap <= args.profile_interpolate_gap_rows + 1:
                for fill_y in range(prev_y + 1, y):
                    t = (fill_y - prev_y) / (y - prev_y)
                    fill_x = (1.0 - t) * prev_x + t * x
                    path.append((fill_x, float(fill_y)))

        path.append((x, float(y)))
        prev_x = x
        prev_y = int(y)

    return np.array(path, dtype=float)


def trend_profile_path_from_mask(mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        empty = np.empty((0, 2), dtype=float)
        return empty, {"x_min": np.array([]), "x_max": np.array([]), "rules": []}

    path: list[tuple[float, float]] = []
    selected: list[float] = []
    x_mins: list[float] = []
    x_maxs: list[float] = []
    rules: list[str] = []

    def recent_trend() -> float:
        if len(selected) < 2:
            return 0.0
        lookback = min(args.trend_lookback_rows, len(selected) - 1)
        previous = np.array(selected[-lookback - 1 :], dtype=float)
        weights = np.linspace(0.5, 1.0, len(previous))
        # Estimate direction from the weighted first-to-last movement, not a
        # smoothed absolute position. This keeps true sharp jumps visible.
        return float((previous[-1] - previous[0]) / max(lookback, 1) * weights[-1])

    for y in rows:
        xs = np.where(mask[y])[0]
        x_min = float(xs.min())
        x_max = float(xs.max())
        center = float(np.median(xs))
        width = x_max - x_min + 1.0

        if width >= args.trend_wide_row_px:
            trend = recent_trend()
            if trend > args.trend_min_slope_px:
                x = x_max
                rule = "trend_xmax"
            elif trend < -args.trend_min_slope_px:
                x = x_min
                rule = "trend_xmin"
            elif selected:
                # If direction is ambiguous, keep the profile stable by using
                # the endpoint nearest to the preceding selected point.
                prev = selected[-1]
                if abs(x_max - prev) < abs(x_min - prev):
                    x = x_max
                    rule = "nearest_xmax"
                else:
                    x = x_min
                    rule = "nearest_xmin"
            else:
                x = center
                rule = "wide_center"
        else:
            x = center
            rule = "center"

        path.append((x, float(y)))
        selected.append(x)
        x_mins.append(x_min)
        x_maxs.append(x_max)
        rules.append(rule)

    return (
        np.array(path, dtype=float),
        {"x_min": np.array(x_mins, dtype=float), "x_max": np.array(x_maxs, dtype=float), "rules": rules},
    )


def x_profile_path_from_mask(mask: np.ndarray, calib: Calibration, origin: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    ys_all, xs_all = np.where(mask)
    if len(xs_all) == 0:
        return np.empty((0, 2), dtype=float)

    x0, y0 = origin
    path: list[tuple[float, float]] = []
    guide_tol = float(getattr(args, "point_guide_tol_y", 2.0))
    previous_y: float | None = None
    previous_x: int | None = None

    for local_x in sorted(set(int(value) for value in xs_all.tolist())):
        local_ys = ys_all[xs_all == local_x]
        global_x = np.full(len(local_ys), x0 + local_x, dtype=float)
        global_y = y0 + local_ys.astype(float)
        x_values, y_values = calib.pixel_to_data(global_x, global_y)
        expected = guide_y_at_x(args, float(x_values[0]))
        if expected is not None:
            distances = np.abs(y_values - expected)
            allowed = distances <= guide_tol
            if np.any(allowed):
                allowed_indices = np.where(allowed)[0]
                candidate_idx = int(allowed_indices[np.argmin(distances[allowed])])
            else:
                candidate_idx = int(np.argmin(distances))
        elif previous_y is not None:
            candidate_idx = int(np.argmin(np.abs(local_ys.astype(float) - previous_y)))
        else:
            candidate_idx = int(np.argmin(np.abs(local_ys.astype(float) - np.median(local_ys))))

        selected_y = float(local_ys[candidate_idx])
        if previous_y is not None and previous_x is not None:
            gap = int(local_x - previous_x)
            max_gap = int(getattr(args, "x_profile_interpolate_gap_px", 0))
            if 1 < gap <= max_gap:
                for fill_x in range(previous_x + 1, local_x):
                    t = (fill_x - previous_x) / gap
                    fill_y = (1.0 - t) * previous_y + t * selected_y
                    path.append((float(fill_x), float(fill_y)))

        path.append((float(local_x), selected_y))
        previous_y = selected_y
        previous_x = local_x

    return np.array(path, dtype=float)


def extract_segments(
    rgb: np.ndarray,
    calib: Calibration,
    args: argparse.Namespace,
) -> tuple[np.ndarray, tuple[int, int], list[CurveSegment]]:
    x0, y0, x1, y1 = calib.crop_box
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
    mask = curve_mask(crop, calib, args)
    mask = apply_exclude_rects(mask, calib, args)
    mask = apply_dash_bridge(mask, args)
    accepted_mask = np.zeros_like(mask, dtype=bool)

    segments: list[CurveSegment] = []
    if args.trace_mode == "x-profile":
        component_iterable = [mask]
    elif args.profile_global_mask and args.trace_mode in ("profile", "trend-profile"):
        component_iterable = [mask]
    else:
        component_iterable = component_masks(mask, args.min_component_area)
    for comp in component_iterable:
        skel = skeleton_largest_component(comp)
        local_aux: list[dict[str, object] | None]
        if args.trace_mode == "trend-profile":
            local_path, aux = trend_profile_path_from_mask(comp, args)
            local_paths = [local_path]
            local_aux = [aux]
        elif args.trace_mode == "profile":
            local_paths = [profile_path_from_mask(comp, args)]
            local_aux = [None]
        elif args.trace_mode == "x-profile":
            local_paths = [x_profile_path_from_mask(comp, calib, (x0, y0), args)]
            local_aux = [None]
        elif args.trace_mode == "longest":
            local_paths = [longest_path(skel)]
            local_aux = [None]
        elif args.trace_mode == "continuous":
            local_paths = [continuous_edge_walk(skel)]
            local_aux = [None]
        else:
            local_paths = trace_skeleton_paths(skel)
            local_aux = [None] * len(local_paths)
        component_used = False
        for local_path, aux in zip(local_paths, local_aux):
            if len(local_path) < args.min_path_points:
                continue

            global_x = local_path[:, 0] + x0
            global_y = local_path[:, 1] + y0
            x_data, y_data = calib.pixel_to_data(global_x, global_y)
            valid = (
                (x_data >= min(calib.x_min, calib.x_max) - args.data_tolerance)
                & (x_data <= max(calib.x_min, calib.x_max) + args.data_tolerance)
                & (y_data >= min(calib.y_min, calib.y_max) - args.data_tolerance)
                & (y_data <= max(calib.y_min, calib.y_max) + args.data_tolerance)
            )
            if valid.sum() < args.min_path_points:
                continue
            x_min_data = None
            x_max_data = None
            selection_rule = None
            if aux is not None:
                aux_x_min = np.asarray(aux["x_min"], dtype=float)
                aux_x_max = np.asarray(aux["x_max"], dtype=float)
                aux_rules = list(aux["rules"])
                if len(aux_x_min) == len(local_path) and len(aux_x_max) == len(local_path):
                    global_x_min = aux_x_min + x0
                    global_x_max = aux_x_max + x0
                    x_min_data, _ = calib.pixel_to_data(global_x_min[valid], global_y[valid])
                    x_max_data, _ = calib.pixel_to_data(global_x_max[valid], global_y[valid])
                    x_min_data = np.clip(x_min_data, min(calib.x_min, calib.x_max), max(calib.x_min, calib.x_max))
                    x_max_data = np.clip(x_max_data, min(calib.x_min, calib.x_max), max(calib.x_min, calib.x_max))
                    selection_rule = [rule for rule, keep in zip(aux_rules, valid.tolist()) if keep]
            segments.append(
                CurveSegment(
                    segment_id=len(segments) + 1,
                    pixel_path=np.column_stack([global_x[valid], global_y[valid]]),
                    x=np.clip(x_data[valid], min(calib.x_min, calib.x_max), max(calib.x_min, calib.x_max)),
                    y=np.clip(y_data[valid], min(calib.y_min, calib.y_max), max(calib.y_min, calib.y_max)),
                    x_min=x_min_data,
                    x_max=x_max_data,
                    selection_rule=selection_rule,
                )
            )
            component_used = True
        if component_used:
            accepted_mask |= comp

    if not segments:
        raise RuntimeError("No curve segments extracted. Try --curve-preset or lower --min-component-area.")
    return accepted_mask, (x0, y0), segments


def pick_font(candidates: list[str]) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def write_csvs(out_dir: Path, stem: str, segments: list[CurveSegment]) -> tuple[Path, Path]:
    csv_path = out_dir / f"{stem}_digitized.csv"
    gap_csv_path = out_dir / f"{stem}_excel_gap.csv"
    has_aux = any(segment.x_min is not None and segment.x_max is not None for segment in segments)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        header = ["segment_id", "point_order", "x_value", "y_value", "pixel_x", "pixel_y"]
        if has_aux:
            header.extend(["x_min_value", "x_max_value", "selection_rule"])
        writer.writerow(header)
        for segment in segments:
            for idx, ((px, py), x, y) in enumerate(zip(segment.pixel_path, segment.x, segment.y)):
                row = [segment.segment_id, idx + 1, f"{x:.6f}", f"{y:.6f}", f"{px:.3f}", f"{py:.3f}"]
                if has_aux:
                    if segment.x_min is not None and segment.x_max is not None and segment.selection_rule is not None:
                        row.extend([f"{segment.x_min[idx]:.6f}", f"{segment.x_max[idx]:.6f}", segment.selection_rule[idx]])
                    else:
                        row.extend(["", "", ""])
                writer.writerow(row)

    with gap_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_value", "y_value", "segment_id", "point_order"])
        for segment in segments:
            for order, (x, y) in enumerate(zip(segment.x, segment.y), start=1):
                writer.writerow([f"{x:.6f}", f"{y:.6f}", segment.segment_id, order])
            writer.writerow(["", "", "", ""])

    return csv_path, gap_csv_path


def draw_validation(
    rgb: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int],
    segments: list[CurveSegment],
    out_dir: Path,
    stem: str,
) -> tuple[Path, Path, Path]:
    x0, y0 = origin
    red_mask = Image.new("RGB", (rgb.shape[1], rgb.shape[0]), "white")
    mask_pixels = np.array(red_mask)
    ys, xs = np.where(mask)
    mask_pixels[ys + y0, xs + x0] = np.array([192, 0, 0], dtype=np.uint8)
    red_mask = Image.fromarray(mask_pixels)

    skeleton_overlay = Image.fromarray(rgb).convert("RGBA")
    skeleton_overlay = Image.alpha_composite(skeleton_overlay, Image.new("RGBA", skeleton_overlay.size, (255, 255, 255, 105)))
    draw = ImageDraw.Draw(skeleton_overlay)
    redraw_overlay = Image.fromarray(rgb).convert("RGBA")
    redraw_draw = ImageDraw.Draw(redraw_overlay)
    for segment in segments:
        points = [tuple(map(float, xy)) for xy in segment.pixel_path]
        draw.line(points, fill="#00a6ffff", width=2, joint="curve")
        redraw_draw.line(points, fill="#00c853ff", width=2, joint="curve")

    mask_path = out_dir / f"{stem}_red_mask.png"
    skeleton_path = out_dir / f"{stem}_skeleton_overlay.png"
    overlay_path = out_dir / f"{stem}_redraw_overlay.png"
    red_mask.save(mask_path)
    skeleton_overlay.convert("RGB").save(skeleton_path)
    redraw_overlay.convert("RGB").save(overlay_path)
    return mask_path, skeleton_path, overlay_path


def data_x_to_pixel(calib: Calibration, x_value: float) -> float:
    return calib.left + (x_value - calib.x_min) / (calib.x_max - calib.x_min) * (calib.right - calib.left)


def render_segments_mask(
    mask_shape: tuple[int, int],
    origin: tuple[int, int],
    segments: list[CurveSegment],
    args: argparse.Namespace,
) -> np.ndarray:
    x0, y0 = origin
    render = np.zeros(mask_shape, dtype=np.uint8)
    for segment in segments:
        points = np.array(
            [[int(round(px - x0)), int(round(py - y0))] for px, py in segment.pixel_path],
            dtype=np.int32,
        )
        if len(points) > 1:
            cv2.polylines(
                render,
                [points],
                isClosed=False,
                color=255,
                thickness=args.react_render_width_px,
                lineType=cv2.LINE_AA,
            )
        elif len(points) == 1:
            cv2.circle(render, tuple(points[0]), radius=max(1, args.react_render_width_px // 2), color=255, thickness=-1)
    if args.react_dilation_iterations > 0:
        render = cv2.dilate(
            (render > 0).astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=args.react_dilation_iterations,
        )
    return render > 0


def react_component_decision(area: int, width: int, height: int, args: argparse.Namespace) -> tuple[bool, str]:
    ratio = width / max(height, 1)
    if area >= args.react_large_area_px:
        return True, "large_area"
    if area >= args.react_min_area_px and width >= args.react_min_width_px and ratio >= args.react_horizontal_ratio:
        return True, "horizontal_residual"
    return False, "small_or_nonhorizontal_residual"


def duplicate_depth_count(segments: list[CurveSegment]) -> int:
    values = [round(float(y), 6) for segment in segments for y in segment.y.tolist()]
    return len(values) - len(set(values))


def write_dict_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            formatted = row.copy()
            for key, value in list(formatted.items()):
                if isinstance(value, float):
                    formatted[key] = f"{value:.6f}"
            writer.writerow(formatted)


def single_profile_react(
    mask: np.ndarray,
    origin: tuple[int, int],
    calib: Calibration,
    segments: list[CurveSegment],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], np.ndarray]:
    x0, y0 = origin
    base_coverage = render_segments_mask(mask.shape, origin, segments, args)
    residual_before = mask & ~base_coverage
    n, labels, stats, _ = cv2.connectedComponentsWithStats(residual_before.astype(np.uint8), 8)

    row_map: dict[int, list[tuple[CurveSegment, int]]] = {}
    for segment in segments:
        for idx, (_px, py) in enumerate(segment.pixel_path):
            local_y = int(round(float(py) - y0))
            row_map.setdefault(local_y, []).append((segment, idx))

    component_rows: list[dict[str, object]] = []
    corrections: list[dict[str, object]] = []
    candidate_counter = 1

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < args.react_min_area_px:
            continue
        bbox_x = int(stats[label, cv2.CC_STAT_LEFT])
        bbox_y = int(stats[label, cv2.CC_STAT_TOP])
        bbox_width = int(stats[label, cv2.CC_STAT_WIDTH])
        bbox_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        accepted, reason = react_component_decision(area, bbox_width, bbox_height, args)
        candidate_id = f"react_{candidate_counter:03d}"
        candidate_counter += 1

        ys, xs = np.where(labels == label)
        x_data, y_data = calib.pixel_to_data(xs.astype(float) + x0, ys.astype(float) + y0)
        component_rows.append(
            {
                "candidate_id": candidate_id,
                "area_px": area,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "x_min_value": float(np.min(x_data)),
                "x_max_value": float(np.max(x_data)),
                "y_min_value": float(np.min(y_data)),
                "y_max_value": float(np.max(y_data)),
                "accepted": "yes" if accepted else "no",
                "decision_reason": reason,
            }
        )
        if not accepted:
            continue

        for local_y in sorted(set(ys.astype(int).tolist())):
            if local_y not in row_map:
                continue
            row_xs = np.where(labels[local_y] == label)[0].astype(float)
            if len(row_xs) == 0:
                continue
            residual_center = float(np.median(row_xs))
            segment, idx = min(
                row_map[local_y],
                key=lambda item: abs(float(item[0].pixel_path[item[1], 0] - x0) - residual_center),
            )
            old_px = float(segment.pixel_path[idx, 0])
            old_local_x = old_px - x0
            old_x = float(segment.x[idx])

            residual_target = float(np.quantile(row_xs, 0.05 if residual_center < old_local_x else 0.95))
            target_source = "residual_left" if residual_center < old_local_x else "residual_right"
            if segment.x_min is not None and segment.x_max is not None:
                min_px = data_x_to_pixel(calib, float(segment.x_min[idx])) - x0
                max_px = data_x_to_pixel(calib, float(segment.x_max[idx])) - x0
                if abs(min_px - residual_target) <= abs(max_px - residual_target):
                    target_local_x = float(min_px)
                    target_source = "x_min_value"
                else:
                    target_local_x = float(max_px)
                    target_source = "x_max_value"
            else:
                target_local_x = residual_target

            if abs(target_local_x - old_local_x) < args.react_min_shift_px:
                continue

            new_px = float(np.clip(target_local_x + x0, calib.left, calib.right))
            new_x, _ = calib.pixel_to_data(np.array([new_px], dtype=float), np.array([segment.pixel_path[idx, 1]], dtype=float))
            segment.pixel_path[idx, 0] = new_px
            segment.x[idx] = float(new_x[0])
            if segment.selection_rule is not None and idx < len(segment.selection_rule):
                segment.selection_rule[idx] = f"single_react_{target_source}"

            corrections.append(
                {
                    "candidate_id": candidate_id,
                    "segment_id": segment.segment_id,
                    "point_order": idx + 1,
                    "row_local_y": local_y,
                    "y_value": float(segment.y[idx]),
                    "old_x_value": old_x,
                    "new_x_value": float(segment.x[idx]),
                    "old_pixel_x": old_px,
                    "new_pixel_x": new_px,
                    "target_source": target_source,
                    "residual_row_width_px": int(row_xs.max() - row_xs.min() + 1),
                }
            )

    reacted_coverage = render_segments_mask(mask.shape, origin, segments, args)
    residual_after = mask & ~reacted_coverage
    summary = {
        "mask_pixels": int(mask.sum()),
        "base_covered_pixels": int((mask & base_coverage).sum()),
        "base_residual_pixels": int(residual_before.sum()),
        "base_residual_ratio": float(residual_before.sum() / max(mask.sum(), 1)),
        "accepted_candidate_count": sum(1 for row in component_rows if row["accepted"] == "yes"),
        "corrected_row_count": len(corrections),
        "reacted_covered_pixels": int((mask & reacted_coverage).sum()),
        "reacted_residual_pixels": int(residual_after.sum()),
        "reacted_residual_ratio": float(residual_after.sum() / max(mask.sum(), 1)),
        "residual_pixel_reduction": int(residual_before.sum()) - int(residual_after.sum()),
        "duplicate_y_count": duplicate_depth_count(segments),
    }
    return component_rows, corrections, summary, residual_after


def write_single_profile_react_artifacts(
    rgb: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int],
    segments: list[CurveSegment],
    components: list[dict[str, object]],
    corrections: list[dict[str, object]],
    summary: dict[str, object],
    residual_after: np.ndarray,
    out_dir: Path,
    stem: str,
) -> list[Path]:
    x0, y0 = origin
    components_path = out_dir / f"{stem}_single_profile_react_components.csv"
    corrections_path = out_dir / f"{stem}_single_profile_react_corrections.csv"
    summary_path = out_dir / f"{stem}_single_profile_react_summary.csv"
    overlay_path = out_dir / f"{stem}_single_profile_react_overlay.png"

    write_dict_csv(
        components_path,
        components,
        [
            "candidate_id",
            "area_px",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "x_min_value",
            "x_max_value",
            "y_min_value",
            "y_max_value",
            "accepted",
            "decision_reason",
        ],
    )
    write_dict_csv(
        corrections_path,
        corrections,
        [
            "candidate_id",
            "segment_id",
            "point_order",
            "row_local_y",
            "y_value",
            "old_x_value",
            "new_x_value",
            "old_pixel_x",
            "new_pixel_x",
            "target_source",
            "residual_row_width_px",
        ],
    )
    write_dict_csv(summary_path, [summary], list(summary.keys()))

    overlay = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.alpha_composite(overlay, Image.new("RGBA", overlay.size, (255, 255, 255, 115)))
    draw = ImageDraw.Draw(overlay)
    for segment in segments:
        points = [tuple(map(float, xy)) for xy in segment.pixel_path]
        if len(points) > 1:
            draw.line(points, fill="#00a6ffff", width=2, joint="curve")
    corrected = {(int(row["segment_id"]), int(row["point_order"])) for row in corrections}
    for segment in segments:
        for order, (px, py) in enumerate(segment.pixel_path, start=1):
            if (segment.segment_id, order) in corrected:
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 210, 0, 220))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(residual_after.astype(np.uint8), 8)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        show_residual = area >= 75 or (area >= 30 and width >= 12 and width >= 1.5 * max(height, 1))
        if not show_residual:
            continue
        ys, xs = np.where(labels == label)
        for local_x, local_y in zip(xs.tolist(), ys.tolist()):
            draw.point((x0 + local_x, y0 + local_y), fill=(255, 112, 0, 220))
    overlay.convert("RGB").save(overlay_path)
    return [components_path, corrections_path, summary_path, overlay_path]


def preview_color_masks(
    rgb: np.ndarray,
    calib: Calibration,
    out_dir: Path,
    stem: str,
    args: argparse.Namespace,
) -> list[Path]:
    presets = [part.strip() for part in args.preview_presets.split(",") if part.strip()]
    if getattr(args, "color_center", None) and "custom" not in presets:
        presets.append("custom")
    x0, y0, x1, y1 = calib.crop_box
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
    outputs: list[Path] = []
    report_rows: list[list[str | int]] = [["preset", "mask_pixels", "component_count", "large_component_count", "preview_path"]]

    for preset in presets:
        if preset == "custom":
            mask = curve_mask(crop, calib, args)
        else:
            mask = apply_roi(color_mask(crop, preset), calib, args)
        mask = apply_exclude_rects(mask, calib, args)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        component_count = max(n - 1, 0)
        large_component_count = sum(
            1 for label in range(1, n) if int(stats[label, cv2.CC_STAT_AREA]) >= args.min_component_area
        )

        base = Image.fromarray(rgb).convert("RGBA")
        shade = Image.new("RGBA", base.size, (255, 255, 255, 135))
        overlay = Image.alpha_composite(base, shade)
        overlay_arr = np.array(overlay)
        ys, xs = np.where(mask)
        overlay_arr[ys + y0, xs + x0] = np.array([255, 0, 0, 255], dtype=np.uint8)
        preview = Image.fromarray(overlay_arr)
        draw = ImageDraw.Draw(preview)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, 255), width=2)
        draw.text((x0 + 4, max(0, y0 - 18)), f"{preset}: pixels={int(mask.sum())}, large={large_component_count}", fill=(0, 0, 0, 255))

        out = out_dir / f"{stem}_mask_preview_{preset}.png"
        preview.convert("RGB").save(out)
        outputs.append(out)
        report_rows.append([preset, int(mask.sum()), component_count, large_component_count, str(out)])

    report_path = out_dir / f"{stem}_mask_preview_report.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(report_rows)
    outputs.append(report_path)
    return outputs


def axis_ticks(min_value: float, max_value: float, major_unit: float | None, explicit: list[float] | None) -> list[float] | None:
    if explicit:
        return explicit
    if not major_unit:
        return None
    start = math.ceil(min_value / major_unit) * major_unit
    values = []
    value = start
    while value <= max_value + major_unit * 0.001:
        values.append(value)
        value += major_unit
    return values


def save_redraw(
    rgb_shape: tuple[int, int, int],
    calib: Calibration,
    segments: list[CurveSegment],
    out_dir: Path,
    stem: str,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    latin_font = args.font or pick_font(["Arial", "DejaVu Sans"])
    cjk_font = args.cjk_font or pick_font(["Microsoft YaHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans"])

    plt.rcParams.update({"font.family": latin_font, "axes.unicode_minus": False, "figure.dpi": 100, "savefig.dpi": args.dpi})
    height, width = rgb_shape[:2]
    fig = plt.figure(figsize=(width / 100, height / 100), facecolor="white")
    ax = fig.add_axes([
        calib.left / width,
        (height - calib.bottom) / height,
        (calib.right - calib.left) / width,
        (calib.bottom - calib.top) / height,
    ])

    for segment in segments:
        ax.plot(
            segment.x,
            segment.y,
            color=args.line_color,
            linewidth=args.line_width,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )

    ax.set_xlim(calib.x_min, calib.x_max)
    ax.set_ylim(calib.y_max, calib.y_min) if calib.reverse_y else ax.set_ylim(calib.y_min, calib.y_max)
    xticks = axis_ticks(calib.x_min, calib.x_max, args.major_x, parse_list(args.x_ticks))
    yticks = axis_ticks(calib.y_min, calib.y_max, args.major_y, parse_list(args.y_ticks))
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if args.minor_x:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(args.minor_x))
    if args.minor_y:
        ax.yaxis.set_minor_locator(plt.MultipleLocator(args.minor_y))

    if args.x_axis_position == "top":
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    else:
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")
        ax.tick_params(axis="x", bottom=True, labelbottom=True, top=False, labeltop=False)

    ax.tick_params(axis="both", which="major", direction="inout", length=args.major_tick_len, width=args.axis_width, pad=args.tick_pad)
    ax.tick_params(axis="both", which="minor", direction="inout", length=args.minor_tick_len, width=max(args.axis_width * 0.75, 0.6))
    ax.tick_params(axis="y", right=False, labelright=False, left=True, labelleft=True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(latin_font)
        label.set_fontsize(args.tick_font_size)
        label.set_fontweight("bold" if args.bold_ticks else "normal")

    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(args.axis_width)

    if args.x_label:
        ax.set_xlabel(args.x_label, fontname=cjk_font, fontsize=args.label_font_size, labelpad=args.label_pad)
    if args.y_label:
        ax.set_ylabel(args.y_label, fontname=cjk_font, fontsize=args.label_font_size, labelpad=args.label_pad)
        ax.yaxis.set_label_coords(args.y_label_x, 0.5)
    ax.grid(False)

    png = out_dir / f"{stem}_redrawn.png"
    original_png = out_dir / f"{stem}_redrawn_original_size.png"
    svg = out_dir / f"{stem}_redrawn.svg"
    pdf = out_dir / f"{stem}_redrawn.pdf"
    fig.savefig(png, facecolor="white", dpi=args.dpi)
    fig.savefig(original_png, facecolor="white", dpi=100)
    fig.savefig(svg, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, original_png, svg, pdf


def continuity_metrics(segments: list[CurveSegment], args: argparse.Namespace) -> dict[str, float | int | str]:
    lengths = [len(segment.x) for segment in segments]
    total = sum(lengths)
    largest_ratio = max(lengths) / total if total else 0.0
    short_segments = sum(1 for length in lengths if length < args.short_segment_points)
    internal_gap_distances: list[float] = []

    for segment in segments:
        if len(segment.pixel_path) < 2:
            continue
        if args.trace_mode in ("profile", "trend-profile"):
            row_steps = np.abs(np.diff(segment.pixel_path[:, 1]))
            internal_gap_distances.extend(float(step) for step in row_steps if step > args.profile_max_gap_rows)
        else:
            steps = np.linalg.norm(np.diff(segment.pixel_path, axis=0), axis=1)
            internal_gap_distances.extend(float(step) for step in steps if step > args.continuity_gap_px)

    max_gap = max(internal_gap_distances) if internal_gap_distances else 0.0
    fragmented = len(segments) > args.max_continuity_segments and largest_ratio < args.min_continuous_ratio
    has_jumps = len(internal_gap_distances) > 0
    status = "PASS" if not fragmented and not has_jumps else "CHECK"
    return {
        "status": status,
        "segment_count": len(segments),
        "largest_ratio": largest_ratio,
        "short_segments": short_segments,
        "internal_gap_count": len(internal_gap_distances),
        "max_internal_gap": max_gap,
    }


def profile_metrics(segments: list[CurveSegment], mask: np.ndarray, origin: tuple[int, int], args: argparse.Namespace) -> dict[str, float | int | str]:
    if not segments:
        return {
            "status": "CHECK",
            "duplicate_depth_rows": 0,
            "selected_rows": 0,
            "mask_rows": 0,
            "row_coverage": 0.0,
            "max_row_gap": 0.0,
            "p95_dx": 0.0,
            "p95_second_diff": 0.0,
        }

    x0, y0 = origin
    all_rows: list[int] = []
    all_x: list[float] = []
    for segment in segments:
        local = np.round(segment.pixel_path - np.array([[x0, y0]])).astype(int)
        valid = (local[:, 1] >= 0) & (local[:, 1] < mask.shape[0])
        all_rows.extend(local[valid, 1].tolist())
        all_x.extend(segment.pixel_path[valid, 0].astype(float).tolist())

    mask_rows_set = set(np.where(mask.any(axis=1))[0].astype(int).tolist())
    if not all_rows:
        selected_rows = 0
        selected_mask_rows = 0
        duplicate_rows = 0
        max_gap = 0
        p95_dx = 0.0
        p95_second = 0.0
    else:
        rows_arr = np.array(all_rows, dtype=int)
        x_arr = np.array(all_x, dtype=float)
        order = np.lexsort((x_arr, rows_arr))
        rows_arr = rows_arr[order]
        x_arr = x_arr[order]
        unique_rows, counts = np.unique(rows_arr, return_counts=True)
        selected_rows = int(len(unique_rows))
        selected_mask_rows = int(sum(1 for row in unique_rows.tolist() if int(row) in mask_rows_set))
        duplicate_rows = int((counts > 1).sum())
        gaps = np.diff(unique_rows)
        max_gap = int(gaps.max()) if len(gaps) else 0

        # For roughness, collapse duplicate rows with their mean x before
        # measuring first and second differences.
        mean_x_by_row = np.array([x_arr[rows_arr == row].mean() for row in unique_rows])
        dx = np.abs(np.diff(mean_x_by_row))
        second = np.abs(np.diff(mean_x_by_row, n=2))
        p95_dx = float(np.percentile(dx, 95)) if len(dx) else 0.0
        p95_second = float(np.percentile(second, 95)) if len(second) else 0.0

    mask_rows = int(mask.any(axis=1).sum())
    row_coverage = selected_mask_rows / mask_rows if mask_rows else 0.0
    status = "PASS"
    if duplicate_rows > 0 or max_gap > args.profile_max_gap_rows or row_coverage < args.min_profile_row_coverage:
        status = "CHECK"
    if args.max_profile_second_diff_px is not None and p95_second > args.max_profile_second_diff_px:
        status = "CHECK"

    return {
        "status": status,
        "duplicate_depth_rows": duplicate_rows,
        "selected_rows": selected_rows,
        "mask_rows": mask_rows,
        "row_coverage": row_coverage,
        "max_row_gap": max_gap,
        "p95_dx": p95_dx,
        "p95_second_diff": p95_second,
    }


def selection_rule_summary(segments: list[CurveSegment]) -> str:
    counts: dict[str, int] = {}
    for segment in segments:
        if not segment.selection_rule:
            continue
        for rule in segment.selection_rule:
            counts[rule] = counts.get(rule, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{name}={counts[name]}" for name in sorted(counts))


def validation_report(
    calib: Calibration,
    mask: np.ndarray,
    origin: tuple[int, int],
    segments: list[CurveSegment],
    args: argparse.Namespace,
) -> list[str]:
    x0, y0 = origin
    skeleton = np.zeros(mask.shape, dtype=np.uint8)
    for segment in segments:
        local = np.round(segment.pixel_path - np.array([[x0, y0]])).astype(int)
        valid = (local[:, 0] >= 0) & (local[:, 0] < mask.shape[1]) & (local[:, 1] >= 0) & (local[:, 1] < mask.shape[0])
        skeleton[local[valid, 1], local[valid, 0]] = 255

    dist = cv2.distanceTransform(np.where(skeleton > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    distances = dist[mask]
    lengths = [len(segment.x) for segment in segments]
    continuity = continuity_metrics(segments, args)
    profile = profile_metrics(segments, mask, origin, args)
    return [
        f"Input image: {args.input}",
        f"Frame pixels: left={calib.left:.3f}, right={calib.right:.3f}, top={calib.top:.3f}, bottom={calib.bottom:.3f}",
        f"Data range: x={calib.x_min:g}..{calib.x_max:g}; y={calib.y_min:g}..{calib.y_max:g}; reverse_y={calib.reverse_y}.",
        f"Curve preset: {args.curve_preset}; trace_mode={args.trace_mode}; visible segments={len(segments)}; points per segment={lengths}; total points={sum(lengths)}.",
        f"Curve-pixel to extracted-centerline distance: mean={float(distances.mean()):.3f} px; p95={float(np.percentile(distances, 95)):.3f} px.",
        f"Continuity: status={continuity['status']}; segments={continuity['segment_count']}; largest_segment_ratio={continuity['largest_ratio']:.3f}; short_segments(<{args.short_segment_points} pts)={continuity['short_segments']}; internal_gaps={continuity['internal_gap_count']}; max_internal_gap={continuity['max_internal_gap']:.3f} {'rows' if args.trace_mode in ('profile', 'trend-profile') else 'px'}.",
        f"Continuity standard: PASS requires no internal jumps and low fragmentation (segments <= {args.max_continuity_segments} or largest_segment_ratio >= {args.min_continuous_ratio:g}).",
        f"Profile single-valuedness: status={profile['status']}; selected_depth_rows={profile['selected_rows']}; curve_mask_rows={profile['mask_rows']}; row_coverage={profile['row_coverage']:.3f}; duplicate_depth_rows={profile['duplicate_depth_rows']}; max_row_gap={profile['max_row_gap']}; p95_row_dx={profile['p95_dx']:.3f} px; p95_second_diff={profile['p95_second_diff']:.3f} px.",
        f"Trend-profile selection rules: {selection_rule_summary(segments)}.",
        f"Profile standard: PASS requires one x per depth row, row_coverage >= {args.min_profile_row_coverage:g}, max_row_gap <= {args.profile_max_gap_rows}, and duplicate_depth_rows = 0. Smoothness is diagnostic unless --max-profile-second-diff-px is set.",
        "Validation: check *_red_mask.png for correct color isolation, then check *_skeleton_overlay.png and *_redraw_overlay.png for centerline alignment. In profile modes, all-pixel p95 can be high when horizontal stroke portions are collapsed to one x per depth.",
        "For depth-profile source curves, prefer --trace-mode trend-profile or --trace-mode profile for final Excel output. Use --trace-mode graph only as a coverage diagnostic because it may fragment the curve.",
    ]


def write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def write_excel(
    path: Path,
    calib: Calibration,
    segments: list[CurveSegment],
    redrawn_png: Path,
    overlay_png: Path,
    report_lines: list[str],
    args: argparse.Namespace,
) -> None:
    workbook = xlsxwriter.Workbook(path)
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    numeric_fmt = workbook.add_format({"num_format": "0.000"})
    note_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    has_aux = any(segment.x_min is not None and segment.x_max is not None for segment in segments)

    data_ws = workbook.add_worksheet("提取数据")
    headers = ["段号", "点序号", args.x_header, args.y_header, "像素X", "像素Y"]
    if has_aux:
        headers.extend(["x_min", "x_max", "selection_rule"])
    data_ws.write_row(0, 0, headers, header_fmt)
    row = 1
    ranges: list[tuple[int, int, int]] = []
    for segment in segments:
        start = row
        for idx, ((px, py), x, y) in enumerate(zip(segment.pixel_path, segment.x, segment.y)):
            data_ws.write_number(row, 0, segment.segment_id)
            data_ws.write_number(row, 1, idx + 1)
            data_ws.write_number(row, 2, float(x), numeric_fmt)
            data_ws.write_number(row, 3, float(y), numeric_fmt)
            data_ws.write_number(row, 4, float(px), numeric_fmt)
            data_ws.write_number(row, 5, float(py), numeric_fmt)
            if has_aux:
                if segment.x_min is not None and segment.x_max is not None and segment.selection_rule is not None:
                    data_ws.write_number(row, 6, float(segment.x_min[idx]), numeric_fmt)
                    data_ws.write_number(row, 7, float(segment.x_max[idx]), numeric_fmt)
                    data_ws.write_string(row, 8, segment.selection_rule[idx])
            row += 1
        ranges.append((segment.segment_id, start, row - 1))
    data_ws.freeze_panes(1, 0)
    data_ws.set_column("A:B", 10)
    data_ws.set_column("C:H", 14)
    data_ws.set_column("I:I", 18)

    chart_ws = workbook.add_worksheet("Excel绘图")
    chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    for segment_id, start, end in ranges:
        chart.add_series(
            {
                "name": f"段{segment_id}",
                "categories": ["提取数据", start, 2, end, 2],
                "values": ["提取数据", start, 3, end, 3],
                "line": {"color": args.line_color, "width": args.excel_line_width},
                "marker": {"type": "none"},
            }
        )
    x_axis = {
        "name": args.x_label,
        "min": calib.x_min,
        "max": calib.x_max,
        "major_tick_mark": "inside",
        "minor_tick_mark": "inside",
        "label_position": "high" if args.x_axis_position == "top" else "low",
        "line": {"color": "black", "width": 1.5},
        "major_gridlines": {"visible": False},
        "num_font": {"name": "Arial", "bold": args.bold_ticks, "size": 10},
    }
    y_axis = {
        "name": args.y_label,
        "min": calib.y_min,
        "max": calib.y_max,
        "reverse": calib.reverse_y,
        "major_tick_mark": "inside",
        "minor_tick_mark": "inside",
        "line": {"color": "black", "width": 1.5},
        "major_gridlines": {"visible": False},
        "num_font": {"name": "Arial", "bold": args.bold_ticks, "size": 10},
        "name_font": {"name": "Microsoft YaHei", "size": 12},
    }
    if args.major_x is not None:
        x_axis["major_unit"] = args.major_x
    if args.minor_x is not None:
        x_axis["minor_unit"] = args.minor_x
    if args.major_y is not None:
        y_axis["major_unit"] = args.major_y
    if args.minor_y is not None:
        y_axis["minor_unit"] = args.minor_y
    chart.set_x_axis(x_axis)
    chart.set_y_axis(y_axis)
    chart.set_plotarea({"border": {"color": "black", "width": 1.5}, "fill": {"color": "white"}})
    chart.set_chartarea({"border": {"none": True}, "fill": {"color": "white"}})
    chart.set_legend({"none": True})
    chart.set_size({"width": args.excel_chart_width, "height": args.excel_chart_height})
    chart_ws.insert_chart("B2", chart)
    chart_ws.write("J2", "说明", header_fmt)
    chart_ws.write("J3", "图表按可见断段拆成多个 series；请用叠加校验图确认数据化曲线是否压在原图中心。", note_fmt)
    chart_ws.set_column("J:J", 38)

    preview_ws = workbook.add_worksheet("PNG重绘与校验")
    preview_ws.write("A1", "重绘 PNG", header_fmt)
    preview_ws.insert_image("A2", str(redrawn_png), {"x_scale": 0.60, "y_scale": 0.60})
    preview_ws.write("H1", "叠加校验图", header_fmt)
    preview_ws.insert_image("H2", str(overlay_png), {"x_scale": 0.60, "y_scale": 0.60})

    report_ws = workbook.add_worksheet("提取报告")
    report_ws.set_column("A:A", 120)
    report_ws.write("A1", "校准与质量检查", header_fmt)
    for idx, line in enumerate(report_lines, start=2):
        report_ws.write(idx - 1, 0, line, note_fmt)

    workbook.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Digitize a colored curve from a plot image and export CSV/XLSX/redraw/validation artifacts.")
    parser.add_argument("--input", required=True, help="Input plot image path.")
    parser.add_argument("--out-dir", help="Output directory. Defaults to <image-stem>_digitized next to the input.")
    parser.add_argument("--stem", help="Output filename stem. Defaults to input image stem.")
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--reverse-y", action="store_true", help="Use when y/depth values increase downward.")
    parser.add_argument("--axes", nargs=4, type=float, metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"), help="Manual frame pixel centers.")
    parser.add_argument("--curve-preset", choices=["red", "blue", "blue-solid", "green", "purple", "dark"], default="red")
    parser.add_argument(
        "--color-center",
        nargs=3,
        type=float,
        action="append",
        metavar=("R", "G", "B"),
        help="Custom RGB color center for single-curve masking. Repeat to include anti-aliased shades.",
    )
    parser.add_argument("--color-space", choices=["rgb", "lab"], default="rgb", help="Distance space for custom --color-center masks.")
    parser.add_argument("--max-color-dist", type=float, default=60.0, help="Maximum custom color distance in RGB/Lab space.")
    parser.add_argument("--min-chroma", type=float, default=18.0, help="Minimum RGB chroma for custom color masks; set 0 for gray targets.")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Data-coordinate ROI to keep inside the calibrated frame before tracing.",
    )
    parser.add_argument(
        "--pixel-roi",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Global pixel ROI to keep inside the calibrated frame before tracing.",
    )
    parser.add_argument(
        "--exclude-rect",
        nargs=4,
        type=float,
        action="append",
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Global pixel rectangle to remove from the color mask. Repeat for labels, legends, or annotations.",
    )
    parser.add_argument("--preview-masks", action="store_true", help="Write candidate mask previews and exit before digitizing.")
    parser.add_argument("--preview-presets", default="red,green,blue,blue-solid,purple,dark", help="Comma-separated presets for --preview-masks.")
    parser.add_argument("--trace-mode", choices=["longest", "graph", "continuous", "profile", "trend-profile", "x-profile"], default="longest")
    parser.add_argument(
        "--guide-point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        help="Data-coordinate guide point for x-profile selection. Repeat along the expected curve.",
    )
    parser.add_argument("--point-guide-tol-y", type=float, default=2.0, help="Allowed y-distance from guide for x-profile candidate selection.")
    parser.add_argument("--x-profile-interpolate-gap-px", type=int, default=0, help="Fill short missing x-profile pixel-column gaps up to this size.")
    parser.add_argument("--min-path-points", type=int, default=3)
    parser.add_argument("--continuity-gap-px", type=float, default=2.5)
    parser.add_argument("--short-segment-points", type=int, default=10)
    parser.add_argument("--max-continuity-segments", type=int, default=3)
    parser.add_argument("--min-continuous-ratio", type=float, default=0.85)
    parser.add_argument("--profile-interpolate-gap-rows", type=int, default=3)
    parser.add_argument("--profile-global-mask", action="store_true", help="For profile modes, select one row representative from the whole target mask instead of per connected component.")
    parser.add_argument("--profile-max-gap-rows", type=int, default=2)
    parser.add_argument("--min-profile-row-coverage", type=float, default=0.90)
    parser.add_argument("--max-profile-second-diff-px", type=float, default=None)
    parser.add_argument("--dash-bridge", action="store_true", help="Morphologically bridge dashed curve masks before profile extraction.")
    parser.add_argument("--dash-bridge-x-px", type=float, default=9.0, help="Horizontal width of the dashed-curve bridge kernel in pixels.")
    parser.add_argument("--dash-bridge-y-px", type=float, default=25.0, help="Vertical height of the dashed-curve bridge kernel in pixels.")
    parser.add_argument("--dash-bridge-iterations", type=int, default=1, help="Number of dashed-curve bridge morphology iterations.")
    parser.add_argument("--trend-wide-row-px", type=float, default=12.0)
    parser.add_argument("--trend-lookback-rows", type=int, default=8)
    parser.add_argument("--trend-min-slope-px", type=float, default=0.6)
    parser.add_argument("--react-single-profile", action="store_true", help="Run residual-driven single-valued profile correction after profile/trend-profile extraction.")
    parser.add_argument("--react-max-passes", type=int, default=1, help="Maximum residual-driven react passes for --react-single-profile. Capped at 2.")
    parser.add_argument("--react-min-area-px", type=int, default=12, help="Minimum residual component area to consider for profile react.")
    parser.add_argument("--react-large-area-px", type=int, default=75, help="Residual components at or above this area are accepted for single-profile react.")
    parser.add_argument("--react-min-width-px", type=int, default=12, help="Minimum residual component width for horizontal react acceptance.")
    parser.add_argument("--react-horizontal-ratio", type=float, default=1.5, help="Minimum residual bbox width/height ratio for horizontal react acceptance.")
    parser.add_argument("--react-min-shift-px", type=float, default=3.0, help="Ignore proposed react corrections smaller than this pixel shift.")
    parser.add_argument("--react-render-width-px", type=int, default=5, help="Stroke width used to render the current extraction when computing residuals.")
    parser.add_argument("--react-dilation-iterations", type=int, default=1, help="Dilation iterations for the rendered extraction coverage mask.")
    parser.add_argument("--min-component-area", type=int, default=80)
    parser.add_argument("--data-tolerance", type=float, default=50.0)
    parser.add_argument("--frame-threshold", type=float, default=70.0)
    parser.add_argument("--frame-column-fraction", type=float, default=0.30)
    parser.add_argument("--frame-row-fraction", type=float, default=0.25)

    parser.add_argument("--x-label", default="")
    parser.add_argument("--y-label", default="深度（米）")
    parser.add_argument("--x-header", default="横坐标")
    parser.add_argument("--y-header", default="纵坐标")
    parser.add_argument("--x-axis-position", choices=["top", "bottom"], default="top")
    parser.add_argument("--x-ticks", help="Comma-separated x tick values.")
    parser.add_argument("--y-ticks", help="Comma-separated y tick values.")
    parser.add_argument("--major-x", type=float, default=None)
    parser.add_argument("--minor-x", type=float, default=None)
    parser.add_argument("--major-y", type=float, default=None)
    parser.add_argument("--minor-y", type=float, default=None)

    parser.add_argument("--line-color", default="#c00000")
    parser.add_argument("--line-width", type=float, default=2.05)
    parser.add_argument("--excel-line-width", type=float, default=2.15)
    parser.add_argument("--axis-width", type=float, default=2.25)
    parser.add_argument("--major-tick-len", type=float, default=8)
    parser.add_argument("--minor-tick-len", type=float, default=4)
    parser.add_argument("--tick-pad", type=float, default=8)
    parser.add_argument("--tick-font-size", type=float, default=12)
    parser.add_argument("--label-font-size", type=float, default=12)
    parser.add_argument("--label-pad", type=float, default=0)
    parser.add_argument("--y-label-x", type=float, default=-0.11)
    parser.add_argument("--bold-ticks", dest="bold_ticks", action="store_true", default=True)
    parser.add_argument("--no-bold-ticks", dest="bold_ticks", action="store_false")
    parser.add_argument("--font")
    parser.add_argument("--cjk-font")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--excel-chart-width", type=int, default=430)
    parser.add_argument("--excel-chart-height", type=int, default=760)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_path.with_name(f"{input_path.stem}_digitized")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or input_path.stem

    rgb = np.array(Image.open(input_path).convert("RGB"))
    calib = detect_axes(rgb, args)
    if args.preview_masks:
        preview_outputs = preview_color_masks(rgb, calib, out_dir, stem, args)
        for path in preview_outputs:
            print(path)
        return

    mask, origin, segments = extract_segments(rgb, calib, args)
    react_paths: list[Path] = []
    react_summaries: list[dict[str, object]] = []
    if args.react_single_profile:
        if args.trace_mode not in ("profile", "trend-profile"):
            raise RuntimeError("--react-single-profile requires --trace-mode profile or --trace-mode trend-profile.")
        react_passes = max(1, min(2, int(args.react_max_passes)))
        for pass_index in range(1, react_passes + 1):
            components, corrections, react_summary, residual_after = single_profile_react(mask, origin, calib, segments, args)
            react_summary["pass_index"] = pass_index
            react_summary["max_passes"] = react_passes
            react_summaries.append(react_summary)
            pass_stem = stem if react_passes == 1 else f"{stem}_react_pass_{pass_index}"
            react_paths.extend(
                write_single_profile_react_artifacts(
                    rgb,
                    mask,
                    origin,
                    segments,
                    components,
                    corrections,
                    react_summary,
                    residual_after,
                    out_dir,
                    pass_stem,
                )
            )
            if react_summary["corrected_row_count"] == 0 or react_summary["residual_pixel_reduction"] <= 0:
                break

    csv_path, gap_csv_path = write_csvs(out_dir, stem, segments)
    redrawn_png, original_png, redrawn_svg, redrawn_pdf = save_redraw(rgb.shape, calib, segments, out_dir, stem, args)
    mask_path, skeleton_path, overlay_path = draw_validation(rgb, mask, origin, segments, out_dir, stem)
    report_lines = validation_report(calib, mask, origin, segments, args)
    if react_summaries:
        report_lines.extend(["", "Single-profile react: enabled."])
        for react_summary in react_summaries:
            report_lines.append(
                "Single-profile react pass "
                f"{react_summary['pass_index']}/{react_summary['max_passes']}: "
                f"accepted_candidates={react_summary['accepted_candidate_count']}; "
                f"corrected_rows={react_summary['corrected_row_count']}; "
                f"residual_ratio={react_summary['base_residual_ratio']:.4f}->{react_summary['reacted_residual_ratio']:.4f}; "
                f"duplicate_y_count={react_summary['duplicate_y_count']}."
            )
        report_lines.append(
            "Single-profile react preserves one selected x per original y row; accepted corrections are listed in *_single_profile_react_corrections.csv."
        )
    report_path = out_dir / f"{stem}_calibration_report.txt"
    write_report(report_path, report_lines)
    excel_path = out_dir / f"{stem}_digitized_redrawn.xlsx"
    write_excel(excel_path, calib, segments, redrawn_png, overlay_path, report_lines, args)

    for path in [
        excel_path,
        csv_path,
        gap_csv_path,
        redrawn_png,
        original_png,
        redrawn_svg,
        redrawn_pdf,
        mask_path,
        skeleton_path,
        overlay_path,
        report_path,
        *react_paths,
    ]:
        print(path)


if __name__ == "__main__":
    main()
