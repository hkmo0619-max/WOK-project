#!/usr/bin/env python3
"""Intel RealSense D435 single-step boundary and riser-height detector.

The two RGB boundaries locate the upper and lower edges of the step's front
face. A candidate is accepted only when its upper, middle, and lower samples
belong to the same depth plane. The vertical 3D Y difference between the two
boundaries is then reported as the step height.

q/ESC: quit, r: reset tracking/height, i: show/hide ROI
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DETECTOR_BUILD = "RGBD848_FACE_HEIGHT_V13"
STATUS_SEARCHING = "SEARCHING FOR TWO RGB BOUNDARIES"
STATUS_DETECTED = "RGB BOUNDARIES DETECTED"


@dataclass(frozen=True)
class DetectorConfig:
    width: int = 848
    height: int = 480
    fps: int = 30
    roi_left: float = 0.20
    roi_top: float = 0.35
    roi_right: float = 0.80
    roi_bottom: float = 1.00
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    gaussian_kernel: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.0
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 40
    hough_min_line_length_px: int = 40
    hough_max_line_gap_px: int = 20
    max_angle_deg: float = 7.0
    merge_y_px: float = 9.0
    min_final_length_ratio: float = 0.08
    min_boundary_gap_px: float = 35.0
    max_boundary_gap_px: float = 220.0
    min_pair_overlap_ratio: float = 0.35
    min_pair_common_width_ratio_of_roi: float = 0.25
    min_lower_boundary_y_ratio: float = 0.50
    floor_band_start_px: int = 8
    floor_band_end_px: int = 40
    floor_min_visible_band_px: int = 16
    floor_side_gap_px: int = 8
    floor_min_side_width_px: int = 16
    floor_vertical_gradient_threshold: float = 30.0
    floor_max_edge_density: float = 0.075
    floor_max_density_over_reference: float = 0.060
    floor_preference_bonus: float = 80.0
    confirm_frames: int = 5
    tracking_tolerance_px: float = 14.0
    smoothing_alpha: float = 0.25
    max_missed_frames: int = 12
    display_line_length_px: float = 160.0
    depth_min_m: float = 0.25
    depth_max_m: float = 1.50
    height_x_samples: int = 48
    height_edge_inset_px: int = 8
    height_patch_radius_px: int = 2
    height_min_valid_samples: int = 30
    height_face_depth_fractions: Tuple[float, ...] = (0.00, 0.50, 1.00)
    height_max_edge_depth_delta_m: float = 0.01
    height_mad_scale: float = 3.5
    height_history_frames: int = 7
    height_history_reset_cm: float = 4.0
    height_min_m: float = 0.02
    height_max_m: float = 0.30


@dataclass
class _Line:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def length_x(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def slope(self) -> float:
        dx = self.x2 - self.x1
        return 0.0 if abs(dx) < 1e-6 else (self.y2 - self.y1) / dx


@dataclass
class BoundaryLine(_Line):
    score: float

    def as_array(self) -> np.ndarray:
        return np.array((self.x1, self.y1, self.x2, self.y2), np.float32)

    @classmethod
    def from_array(cls, values: np.ndarray, score: float = 0.0) -> "BoundaryLine":
        return cls(*(float(v) for v in values), score)


BoundaryPair = Tuple[BoundaryLine, BoundaryLine]
Deproject = Callable[[Tuple[float, float], float], Sequence[float]]


@dataclass(frozen=True)
class HeightMeasurement:
    height_m: float
    raw_height_m: float
    front_distance_m: float
    spread_m: float
    valid_samples: int
    total_samples: int


def roi_pixels(cfg: DetectorConfig) -> Tuple[int, int, int, int]:
    return tuple(
        int(round(value))
        for value in (
            cfg.width * cfg.roi_left,
            cfg.height * cfg.roi_top,
            cfg.width * cfg.roi_right,
            cfg.height * cfg.roi_bottom,
        )
    )


def build_hidden_edge_image(
    color_bgr: np.ndarray, cfg: DetectorConfig
) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(
        cfg.clahe_clip_limit, cfg.clahe_tile_size
    ).apply(gray)
    blurred = cv2.GaussianBlur(
        enhanced, cfg.gaussian_kernel, cfg.gaussian_sigma
    )
    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)
    x0, y0, x1, y1 = roi_pixels(cfg)
    mask = np.zeros_like(edges)
    mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(edges, mask), enhanced


def contrast_across_line(
    gray: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> float:
    left, right = round(min(x1, x2)), round(max(x1, x2))
    if right - left < 8:
        return 0.0
    xs = np.linspace(left, right, min(160, right - left + 1)).astype(np.int32)
    ys = np.rint(y1 + (y2 - y1) / max(x2 - x1, 1e-6) * (xs - x1)).astype(
        np.int32
    )
    upper, lower = [], []
    for offset in (3, 4, 5):
        yu, yl = ys - offset, ys + offset
        valid_u = (yu >= 0) & (yu < gray.shape[0])
        valid_l = (yl >= 0) & (yl < gray.shape[0])
        if np.any(valid_u):
            upper.append(gray[yu[valid_u], xs[valid_u]])
        if np.any(valid_l):
            lower.append(gray[yl[valid_l], xs[valid_l]])
    if not upper or not lower:
        return 0.0
    return abs(float(np.median(np.concatenate(upper))) - float(np.median(np.concatenate(lower))))


def find_horizontal_candidates(
    color_bgr: np.ndarray, cfg: DetectorConfig
) -> List[BoundaryLine]:
    edges, gray = build_hidden_edge_image(color_bgr, cfg)
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_length_px,
        maxLineGap=cfg.hough_max_line_gap_px,
    )
    if raw is None:
        return []

    rx0, _, rx1, _ = roi_pixels(cfg)
    roi_center, roi_width = (rx0 + rx1) / 2, max(1.0, float(rx1 - rx0))
    candidates = []
    for packed in np.asarray(raw).reshape(-1, 4):
        x1, y1, x2, y2 = (float(v) for v in packed)
        if x2 < x1:
            x1, x2, y1, y2 = x2, x1, y2, y1
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        if angle > cfg.max_angle_deg:
            continue
        contrast = contrast_across_line(gray, x1, y1, x2, y2)
        angle_factor = max(0.0, 1.0 - angle / cfg.max_angle_deg)
        center_factor = max(0.35, 1.0 - abs((x1 + x2) / 2 - roi_center) / roi_width)
        contrast_factor = 1.0 + min(contrast / 50.0, 1.0) * 0.30
        score = length * (0.60 + 0.40 * angle_factor) * center_factor * contrast_factor
        candidates.append(BoundaryLine(x1, y1, x2, y2, score))
    return candidates


def merge_similar_lines(
    candidates: Sequence[BoundaryLine], cfg: DetectorConfig
) -> List[BoundaryLine]:
    clusters: List[List[BoundaryLine]] = []
    for line in sorted(candidates, key=lambda item: item.center_y):
        nearby = [
            (abs(line.center_y - np.average(
                [item.center_y for item in cluster],
                weights=[item.score for item in cluster],
            )), cluster)
            for cluster in clusters
        ]
        nearby = [item for item in nearby if item[0] <= cfg.merge_y_px]
        (min(nearby, key=lambda item: item[0])[1] if nearby else clusters.append([line]))
        if nearby:
            min(nearby, key=lambda item: item[0])[1].append(line)

    merged = []
    for cluster in clusters:
        weights = np.array([line.score for line in cluster], np.float64)
        weights /= max(float(weights.sum()), 1e-9)
        center_y = float(sum(line.center_y * weight for line, weight in zip(cluster, weights)))
        slope = float(sum(line.slope * weight for line, weight in zip(cluster, weights)))
        x1, x2 = min(line.x1 for line in cluster), max(line.x2 for line in cluster)
        if x2 - x1 < cfg.width * cfg.min_final_length_ratio:
            continue
        center_x = (x1 + x2) / 2
        y1, y2 = center_y + slope * (x1 - center_x), center_y + slope * (x2 - center_x)
        score = sum(line.score for line in cluster) + 0.25 * (x2 - x1)
        merged.append(BoundaryLine(x1, y1, x2, y2, float(score)))
    return merged


def horizontal_overlap_ratio(a: BoundaryLine, b: BoundaryLine) -> float:
    overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    return overlap / max(1.0, min(a.length_x, b.length_x))


def crop_line_to_x_range(line: BoundaryLine, left: float, right: float) -> BoundaryLine:
    slope = (line.y2 - line.y1) / max(line.x2 - line.x1, 1e-6)
    return BoundaryLine(
        left,
        line.y1 + slope * (left - line.x1),
        right,
        line.y1 + slope * (right - line.x1),
        line.score,
    )


def center_crop_line(
    line: BoundaryLine, max_length_px: Optional[float]
) -> BoundaryLine:
    if not max_length_px or max_length_px <= 0 or line.length_x <= max_length_px:
        return line
    center = (line.x1 + line.x2) / 2
    return crop_line_to_x_range(
        line, center - max_length_px / 2, center + max_length_px / 2
    )


def make_vertical_gradient_image(color_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))


def region_edge_density(
    gradient: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    threshold: float,
) -> Optional[float]:
    height, width = gradient.shape
    x0, x1 = int(np.clip(x0, 0, width)), int(np.clip(x1, 0, width))
    y0, y1 = int(np.clip(y0, 0, height)), int(np.clip(y1, 0, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return float(np.mean(gradient[y0:y1, x0:x1] >= threshold))


def floor_visible_below_pair(
    gradient: np.ndarray, pair: BoundaryPair, cfg: DetectorConfig
) -> bool:
    lower = pair[1]
    roi_x0, _, roi_x1, roi_y1 = roi_pixels(cfg)
    line_y = round(lower.center_y)
    y0 = line_y + cfg.floor_band_start_px
    y1 = min(line_y + cfg.floor_band_end_px, roi_y1, gradient.shape[0])
    if y1 - y0 < cfg.floor_min_visible_band_px:
        return False

    trim = max(4, round(lower.length_x * 0.08))
    density = region_edge_density(
        gradient,
        round(lower.x1) + trim,
        y0,
        round(lower.x2) - trim,
        y1,
        cfg.floor_vertical_gradient_threshold,
    )
    if density is None or density > cfg.floor_max_edge_density:
        return False

    sides = []
    left_end = round(lower.x1) - cfg.floor_side_gap_px
    right_start = round(lower.x2) + cfg.floor_side_gap_px
    side_ranges = []
    if left_end - roi_x0 >= cfg.floor_min_side_width_px:
        side_ranges.append((roi_x0, left_end))
    if roi_x1 - right_start >= cfg.floor_min_side_width_px:
        side_ranges.append((right_start, roi_x1))
    for x0, x1 in side_ranges:
        value = region_edge_density(
            gradient, x0, y0, x1, y1, cfg.floor_vertical_gradient_threshold
        )
        if value is not None:
            sides.append(value)
    return not sides or density <= min(sides) + cfg.floor_max_density_over_reference


def select_final_pair(
    lines: Sequence[BoundaryLine],
    cfg: DetectorConfig,
    vertical_gradient: Optional[np.ndarray] = None,
    depth_m: Optional[np.ndarray] = None,
) -> Optional[BoundaryPair]:
    best_pair, best_score = None, -float("inf")
    roi_center = cfg.width * (cfg.roi_left + cfg.roi_right) / 2
    _, roi_y0, _, roi_y1 = roi_pixels(cfg)
    roi_width = cfg.width * (cfg.roi_right - cfg.roi_left)
    roi_height = max(1.0, float(roi_y1 - roi_y0))
    max_slope_delta = max(1e-6, 2 * np.tan(np.radians(cfg.max_angle_deg)))

    for i, first in enumerate(lines[:-1]):
        for second in lines[i + 1 :]:
            upper, lower = sorted((first, second), key=lambda line: line.center_y)
            gap = lower.center_y - upper.center_y
            overlap = horizontal_overlap_ratio(upper, lower)
            if not cfg.min_boundary_gap_px <= gap <= cfg.max_boundary_gap_px:
                continue
            if lower.center_y < cfg.height * cfg.min_lower_boundary_y_ratio:
                continue
            if overlap < cfg.min_pair_overlap_ratio:
                continue

            left, right = max(upper.x1, lower.x1), min(upper.x2, lower.x2)
            common_width = right - left
            if common_width < roi_width * cfg.min_pair_common_width_ratio_of_roi:
                continue
            pair = (
                crop_line_to_x_range(upper, left, right),
                crop_line_to_x_range(lower, left, right),
            )
            longest = max(upper.length_x, lower.length_x, 1.0)
            length_similarity = min(upper.length_x, lower.length_x) / longest
            center_factor = max(0.0, 1.0 - abs((left + right) / 2 - roi_center) / (cfg.width / 2))
            support_alignment = common_width / longest
            x_alignment = max(0.0, 1.0 - abs(upper.center_x - lower.center_x) / roi_width)
            slope_alignment = max(0.0, 1.0 - abs(upper.slope - lower.slope) / max_slope_delta)
            lower_factor = float(np.clip((lower.center_y - roi_y0) / roi_height, 0.0, 1.0))
            score = (
                upper.score
                + lower.score
                + 180 * overlap
                + 240 * length_similarity
                + 140 * support_alignment
                + 90 * x_alignment
                + 80 * slope_alignment
                + 120 * lower_factor
                + 70 * center_factor
            )
            if vertical_gradient is not None and floor_visible_below_pair(
                vertical_gradient, (upper, lower), cfg
            ):
                score += cfg.floor_preference_bonus
            if depth_m is not None:
                valid, total = face_plane_support(depth_m, pair, cfg)
                if valid < cfg.height_min_valid_samples:
                    continue
                score += 200.0 * valid / max(total, 1)
            if score > best_score:
                best_pair, best_score = pair, score
    return best_pair


def detect_boundary_pair(
    color_bgr: np.ndarray,
    cfg: DetectorConfig,
    depth_m: Optional[np.ndarray] = None,
) -> Optional[BoundaryPair]:
    lines = merge_similar_lines(find_horizontal_candidates(color_bgr, cfg), cfg)
    return select_final_pair(
        lines, cfg, make_vertical_gradient_image(color_bgr), depth_m
    )


class BoundaryTracker:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.pending = self.confirmed = None
        self.consecutive = self.missed = 0

    @staticmethod
    def _pair_array(pair: BoundaryPair) -> np.ndarray:
        return np.stack([line.as_array() for line in pair])

    @staticmethod
    def _center_ys(values: np.ndarray) -> np.ndarray:
        return (values[:, 1] + values[:, 3]) / 2

    def _error(self, current: np.ndarray, saved: np.ndarray) -> float:
        return float(np.max(np.abs(self._center_ys(current) - self._center_ys(saved))))

    def update(self, pair: Optional[BoundaryPair]) -> Optional[BoundaryPair]:
        if pair is None:
            self.missed += 1
            self.pending, self.consecutive = None, 0
            if self.missed > self.cfg.max_missed_frames:
                self.reset()
            return self._confirmed_pair()

        current = self._pair_array(pair)
        if self.confirmed is not None:
            if self._error(current, self.confirmed) <= self.cfg.tracking_tolerance_px:
                alpha = self.cfg.smoothing_alpha
                self.confirmed = (1 - alpha) * self.confirmed + alpha * current
                self.pending = self.confirmed.copy()
                self.consecutive, self.missed = self.cfg.confirm_frames, 0
                return self._confirmed_pair()
            self.missed += 1
        else:
            self.missed = 0

        if self.pending is None or self._error(current, self.pending) > self.cfg.tracking_tolerance_px:
            self.pending, self.consecutive = current, 1
        else:
            alpha = self.cfg.smoothing_alpha
            self.pending = (1 - alpha) * self.pending + alpha * current
            self.consecutive += 1
        if self.consecutive >= self.cfg.confirm_frames:
            self.confirmed, self.missed = self.pending.copy(), 0
        elif self.missed > self.cfg.max_missed_frames:
            self.reset()
        return self._confirmed_pair()

    def _confirmed_pair(self) -> Optional[BoundaryPair]:
        if self.confirmed is None:
            return None
        return tuple(BoundaryLine.from_array(line) for line in self.confirmed)


def pair_pixel_gap(pair: BoundaryPair) -> float:
    return pair[1].center_y - pair[0].center_y


def line_y_at_x(line: BoundaryLine, x: float) -> float:
    return line.y1 + line.slope * (x - line.x1)


def depth_patch_median(
    depth_m: np.ndarray, x: float, y: float, cfg: DetectorConfig
) -> Optional[float]:
    """Return a robust depth from a small patch fully inside the step face."""
    radius = cfg.height_patch_radius_px
    cx, cy = round(x), round(y)
    x0, x1 = max(0, cx - radius), min(depth_m.shape[1], cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(depth_m.shape[0], cy + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[
        np.isfinite(patch)
        & (patch >= cfg.depth_min_m)
        & (patch <= cfg.depth_max_m)
    ]
    return float(np.median(valid)) if valid.size >= 5 else None


def face_column_depth_m(
    depth_m: np.ndarray,
    pair: BoundaryPair,
    x: float,
    cfg: DetectorConfig,
) -> Optional[float]:
    """Return one face depth only when upper/middle/lower samples agree."""
    upper_y = line_y_at_x(pair[0], x)
    lower_y = line_y_at_x(pair[1], x)
    usable_top = upper_y + cfg.height_edge_inset_px
    usable_bottom = lower_y - cfg.height_edge_inset_px
    if usable_bottom - usable_top <= 2 * cfg.height_patch_radius_px:
        return None

    depths = [
        depth_patch_median(
            depth_m,
            x,
            usable_top + fraction * (usable_bottom - usable_top),
            cfg,
        )
        for fraction in cfg.height_face_depth_fractions
    ]
    if any(value is None for value in depths):
        return None
    values = np.asarray(depths, np.float64)
    if float(np.ptp(values)) > cfg.height_max_edge_depth_delta_m + 1e-9:
        return None
    return float(np.median(values))


def face_plane_support(
    depth_m: np.ndarray, pair: BoundaryPair, cfg: DetectorConfig
) -> Tuple[int, int]:
    """Count columns whose face Depth variation is at most 1 cm."""
    left, right = max(pair[0].x1, pair[1].x1), min(pair[0].x2, pair[1].x2)
    trim = max(6.0, (right - left) * 0.08)
    if right - left <= 2 * trim:
        return 0, cfg.height_x_samples
    xs = np.linspace(left + trim, right - trim, cfg.height_x_samples)
    valid = sum(face_column_depth_m(depth_m, pair, x, cfg) is not None for x in xs)
    return valid, len(xs)


def measure_step_face_height(
    depth_m: np.ndarray,
    pair: Optional[BoundaryPair],
    deproject: Deproject,
    cfg: DetectorConfig,
) -> Optional[HeightMeasurement]:
    """Measure the vertical 3D Y difference across one step's front face.

    Upper, middle, and lower inset samples must agree within 1 cm. Their median
    face depth is applied to both actual boundary coordinates, so Z difference
    cannot be added to the reported height.
    """
    if pair is None or depth_m.ndim != 2:
        return None

    upper, lower = pair
    left, right = max(upper.x1, lower.x1), min(upper.x2, lower.x2)
    trim = max(6.0, (right - left) * 0.08)
    if right - left <= 2 * trim:
        return None

    heights, distances = [], []
    xs = np.linspace(left + trim, right - trim, cfg.height_x_samples)
    for x in xs:
        upper_y, lower_y = line_y_at_x(upper, x), line_y_at_x(lower, x)
        if lower_y - upper_y <= 2 * (
            cfg.height_edge_inset_px + cfg.height_patch_radius_px
        ):
            continue

        face_z = face_column_depth_m(depth_m, pair, x, cfg)
        if face_z is None:
            continue

        upper_3d = np.asarray(deproject((float(x), upper_y), face_z), np.float64)
        lower_3d = np.asarray(deproject((float(x), lower_y), face_z), np.float64)
        if upper_3d.size < 2 or lower_3d.size < 2:
            continue
        height = abs(float(lower_3d[1] - upper_3d[1]))
        if np.isfinite(height) and cfg.height_min_m <= height <= cfg.height_max_m:
            heights.append(height)
            distances.append(face_z)

    if len(heights) < cfg.height_min_valid_samples:
        return None

    values = np.asarray(heights)
    median = float(np.median(values))
    deviation = np.abs(values - median)
    mad = float(np.median(deviation))
    if mad > 1e-6:
        keep = deviation <= cfg.height_mad_scale * 1.4826 * mad
        values = values[keep]
        distances = np.asarray(distances)[keep]
    else:
        distances = np.asarray(distances)
    if values.size < cfg.height_min_valid_samples:
        return None

    raw = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - raw)))
    return HeightMeasurement(
        raw, raw, float(np.median(distances)), spread, int(values.size), len(xs)
    )


class HeightTracker:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.history: Deque[float] = deque(maxlen=cfg.height_history_frames)
        self.missed = 0

    def reset(self) -> None:
        self.history.clear()
        self.missed = 0

    def update(
        self, measurement: Optional[HeightMeasurement]
    ) -> Optional[HeightMeasurement]:
        if measurement is None:
            self.missed += 1
            if self.missed > self.cfg.max_missed_frames:
                self.reset()
            return None

        self.missed = 0
        if self.history and abs(
            measurement.raw_height_m - float(np.median(self.history))
        ) > self.cfg.height_history_reset_cm / 100:
            self.history.clear()
        self.history.append(measurement.raw_height_m)
        return replace(measurement, height_m=float(np.median(self.history)))


def _outlined_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    outline: int = 5,
) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_final_result(
    color_bgr: np.ndarray,
    pair: Optional[BoundaryPair],
    status: str = STATUS_SEARCHING,
    max_line_length_px: Optional[float] = None,
    measurement: Optional[HeightMeasurement] = None,
) -> np.ndarray:
    display = color_bgr.copy()
    detail = None
    if pair is None:
        text, color = status, (255, 255, 255)
    else:
        for line, color in zip(pair, ((255, 255, 0), (255, 0, 255))):
            visible = center_crop_line(line, max_line_length_px)
            p1 = round(visible.x1), round(visible.y1)
            p2 = round(visible.x2), round(visible.y2)
            cv2.line(display, p1, p2, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.line(display, p1, p2, color, 4, cv2.LINE_AA)
        color = (0, 255, 0)
        if measurement is None:
            text = "STEP FACE DETECTED | WAITING FOR VALID DEPTH"
            detail = f"PIXEL GAP: {pair_pixel_gap(pair):.1f} px"
        else:
            text = f"STEP HEIGHT: {measurement.height_m * 100:.1f} cm"
            detail = (
                f"FRONT Z: {measurement.front_distance_m * 100:.1f} cm | "
                f"VALID: {measurement.valid_samples}/{measurement.total_samples} | "
                f"SPREAD: {measurement.spread_m * 100:.2f} cm"
            )
    _outlined_text(display, text, (18, 34), 0.66, color)
    if detail:
        _outlined_text(display, detail, (18, 62), 0.55, color, 4)
    return display


def draw_roi_overlay(color_bgr: np.ndarray, cfg: DetectorConfig) -> None:
    x0, y0, x1, y1 = roi_pixels(cfg)
    color = (0, 255, 255)
    cv2.rectangle(color_bgr, (x0, y0), (x1 - 1, y1 - 1), color, 2, cv2.LINE_AA)
    _outlined_text(color_bgr, "RGB DETECTION ROI", (x0 + 8, y0 + 24), 0.55, color, 4)


def run_live() -> None:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is required. Run this file where realsense-viewer works."
        ) from exc

    cfg = DetectorConfig()
    boundary_tracker, height_tracker, show_roi = (
        BoundaryTracker(cfg),
        HeightTracker(cfg),
        True,
    )
    pipeline, stream_cfg = rs.pipeline(), rs.config()
    stream_cfg.enable_stream(
        rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps
    )
    stream_cfg.enable_stream(
        rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps
    )
    profile = pipeline.start(stream_cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align_to_color = rs.align(rs.stream.color)
    print(f"Detector build: {DETECTOR_BUILD}")
    print(f"D435 RGB-D step height: {cfg.width}x{cfg.height}@{cfg.fps} FPS")
    print("q/ESC: quit, r: reset tracking/height, i: show/hide ROI")
    try:
        while True:
            frames = align_to_color.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth_m = (
                np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            )
            pair = detect_boundary_pair(color, cfg, depth_m)
            confirmed = boundary_tracker.update(pair)
            intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

            def deproject(pixel: Tuple[float, float], depth: float) -> Sequence[float]:
                return rs.rs2_deproject_pixel_to_point(intrinsics, list(pixel), depth)

            measurement = height_tracker.update(
                measure_step_face_height(depth_m, confirmed, deproject, cfg)
            )
            display = draw_final_result(
                color,
                confirmed,
                STATUS_DETECTED if confirmed or pair else STATUS_SEARCHING,
                cfg.display_line_length_px,
                measurement,
            )
            if show_roi:
                draw_roi_overlay(display, cfg)
            cv2.imshow("D435 RGB - Final Two Boundaries", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                boundary_tracker.reset()
                height_tracker.reset()
            elif key == ord("i"):
                show_roi = not show_roi
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live()
