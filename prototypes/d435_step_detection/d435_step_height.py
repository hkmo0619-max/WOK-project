#!/usr/bin/env python3
"""D435 two-boundary step-height detector.

Measurement rule used by this file (no plane fitting):

1. Detect a short, reliable upper/lower RGB boundary pair.
2. Sample Depth 8..10 px ABOVE the upper boundary and BELOW the lower one.
3. Deproject every valid sample to SDK 3D XYZ coordinates.
4. height = abs(median(Y_lower) - median(Y_upper)).

Only the RGB result window is shown. Canny/Depth images stay internal.

Keys: q/ESC quit, r reset, i show/hide the detection ROI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Deque, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


BUILD = "D435_TWO_BOUNDARY_MEDIAN_Y_V1_8_TO_10PX"
Deproject = Callable[[Tuple[float, float], float], Sequence[float]]


@dataclass(frozen=True)
class Config:
    # D435 streams
    width: int = 848
    height: int = 480
    fps: int = 30

    # Central/lower RGB detection ROI. Coordinates are never cropped, so the
    # RGB pixels and the aligned Depth pixels keep the same coordinate system.
    roi_left: float = 0.20
    roi_top: float = 0.35
    roi_right: float = 0.80
    roi_bottom: float = 1.00

    # RGB horizontal-line detector
    clahe_clip: float = 2.0
    canny_low: int = 45
    canny_high: int = 135
    hough_threshold: int = 30
    hough_min_length_px: int = 42
    hough_max_gap_px: int = 18
    max_line_angle_deg: float = 7.0
    merge_y_px: float = 7.0
    merge_x_gap_px: float = 28.0
    merge_slope_delta: float = 0.10
    max_merged_lines: int = 24

    # Two-boundary geometry. The complete object width is not required.
    min_boundary_length_px: float = 55.0
    min_common_width_px: float = 70.0
    min_overlap_ratio: float = 0.42
    min_boundary_gap_px: float = 25.0
    max_boundary_gap_px: float = 140.0
    max_pair_slope_delta: float = 0.12
    min_space_below_lower_px: int = 10
    min_lower_position_in_roi: float = 0.36
    display_line_length_px: float = 180.0

    # Valid D435 working range selected for this project.
    depth_min_m: float = 0.30
    depth_max_m: float = 1.00
    depth_patch_radius_px: int = 1

    # Quick Depth check used only to reject printed/background RGB lines.
    # It does NOT calculate height and does NOT fit a plane.
    evidence_x_samples: int = 18
    evidence_offsets_px: Tuple[int, ...] = (6, 10, 14)
    evidence_min_valid_columns: int = 7
    evidence_min_transition_m: float = 0.004
    evidence_max_face_layer_delta_m: float = 0.035
    min_evidence_tier: int = 1

    # Height measurement bands: exactly 8~10 px from each edge.
    height_x_samples: int = 48
    height_band_offsets_px: Tuple[int, ...] = (6, 7, 8)
    height_x_inset_px: float = 7.0
    height_min_valid_columns: int = 24
    height_min_paired_columns: int = 18
    surface_outlier_mad_scale: float = 3.5
    surface_outlier_min_tolerance_m: float = 0.003
    surface_max_y_spread_m: float = 0.012
    height_max_column_spread_m: float = 0.012
    height_min_m: float = 0.010
    height_max_m: float = 0.200
    height_hold_frames: int = 2

    # Boundary and height temporal stability
    boundary_window_frames: int = 5
    boundary_required_frames: int = 3
    boundary_y_tolerance_px: float = 13.0
    boundary_gap_tolerance_px: float = 15.0
    boundary_slope_tolerance: float = 0.08
    boundary_hold_frames: int = 2
    evidence_confirm_frames: int = 2
    evidence_hold_frames: int = 2
    boundary_smoothing_alpha: float = 0.30
    max_missed_frames: int = 6
    height_history_frames: int = 7
    height_required_frames: int = 3
    height_history_reset_m: float = 0.040
    height_max_temporal_spread_m: float = 0.006

    # Spatial/temporal filtering is applied after Depth-to-color alignment.
    use_realsense_filters: bool = True


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
            self.y1, self.y2 = self.y2, self.y1

    @property
    def length_x(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def slope(self) -> float:
        return (self.y2 - self.y1) / max(self.x2 - self.x1, 1e-6)

    def y_at(self, x: float) -> float:
        return self.y1 + self.slope * (x - self.x1)

    def crop(self, left: float, right: float) -> "Line":
        return Line(left, self.y_at(left), right, self.y_at(right), self.score)

    def as_array(self) -> np.ndarray:
        return np.asarray((self.x1, self.y1, self.x2, self.y2), np.float64)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "Line":
        return cls(*(float(value) for value in values), score=0.0)


BoundaryPair = Tuple[Line, Line]


@dataclass(frozen=True)
class PairEvidence:
    tier: int
    top_z_m: float
    face_z_m: float
    floor_z_m: float
    upper_transition_m: float
    lower_transition_m: float
    face_layer_delta_m: float
    top_valid: int
    face_valid: int
    floor_valid: int


@dataclass(frozen=True)
class SurfaceStats:
    median_y_m: float
    median_z_m: float
    spread_y_m: float
    valid_columns: int
    y_by_column: np.ndarray


@dataclass(frozen=True)
class HeightMeasurement:
    height_m: float
    raw_height_m: float
    signed_height_m: float
    upper_y_m: float
    lower_y_m: float
    front_z_m: float
    upper_valid: int
    lower_valid: int
    paired_valid: int
    total_columns: int
    column_spread_m: float
    temporal_spread_m: float = 0.0


def roi_bounds(cfg: Config) -> Tuple[int, int, int, int]:
    return (
        int(round(cfg.width * cfg.roi_left)),
        int(round(cfg.height * cfg.roi_top)),
        int(round(cfg.width * cfg.roi_right)),
        int(round(cfg.height * cfg.roi_bottom)),
    )


def robust_mad(values: np.ndarray) -> float:
    if values.size == 0:
        return float("inf")
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)))


def _outlined_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
) -> None:
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0), 5, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA,
    )


def make_rgb_edges(color_bgr: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(cfg.clahe_clip, (8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)
    )

    x0, y0, x1, y1 = roi_bounds(cfg)
    mask = np.zeros_like(edges)
    mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(edges, mask), gray


def contrast_across_line(gray: np.ndarray, line: Line) -> float:
    count = int(np.clip(round(line.length_x), 12, 120))
    xs = np.linspace(line.x1, line.x2, count)
    upper_values: List[float] = []
    lower_values: List[float] = []
    for x in xs:
        y = line.y_at(float(x))
        xi = int(np.clip(round(x), 0, gray.shape[1] - 1))
        for offset in (3, 5):
            yu, yl = round(y - offset), round(y + offset)
            if 0 <= yu < gray.shape[0]:
                upper_values.append(float(gray[yu, xi]))
            if 0 <= yl < gray.shape[0]:
                lower_values.append(float(gray[yl, xi]))
    if not upper_values or not lower_values:
        return 0.0
    return abs(float(np.median(upper_values)) - float(np.median(lower_values)))


def find_horizontal_lines(color_bgr: np.ndarray, cfg: Config) -> List[Line]:
    edges, gray = make_rgb_edges(color_bgr, cfg)
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        cfg.hough_threshold,
        minLineLength=cfg.hough_min_length_px,
        maxLineGap=cfg.hough_max_gap_px,
    )
    if raw is None:
        return []

    x0, _, x1, _ = roi_bounds(cfg)
    roi_center = (x0 + x1) * 0.5
    roi_half_width = max((x1 - x0) * 0.5, 1.0)
    result: List[Line] = []
    # reshape(-1, 4) handles both OpenCV return layouts: N x 1 x 4 and N x 4.
    for packed in np.asarray(raw).reshape(-1, 4):
        line = Line(*(float(value) for value in packed))
        angle = abs(float(np.degrees(np.arctan2(
            line.y2 - line.y1, line.x2 - line.x1
        ))))
        if angle > cfg.max_line_angle_deg:
            continue
        contrast = contrast_across_line(gray, line)
        angle_factor = 1.0 - angle / max(cfg.max_line_angle_deg, 1e-6)
        center_factor = max(
            0.35, 1.0 - abs(line.center_x - roi_center) / roi_half_width
        )
        line.score = (
            line.length_x
            * (0.65 + 0.35 * angle_factor)
            * center_factor
            * (1.0 + min(contrast / 45.0, 1.0) * 0.35)
        )
        result.append(line)
    return result


def horizontal_gap(a: Line, b: Line) -> float:
    if a.x2 < b.x1:
        return b.x1 - a.x2
    if b.x2 < a.x1:
        return a.x1 - b.x2
    return 0.0


def merge_cluster(lines: Sequence[Line]) -> Line:
    weights = np.asarray([max(line.score, 1.0) for line in lines], np.float64)
    weights /= float(weights.sum())
    anchor_x = float(sum(line.center_x * w for line, w in zip(lines, weights)))
    anchor_y = float(sum(line.y_at(anchor_x) * w for line, w in zip(lines, weights)))
    slope = float(sum(line.slope * w for line, w in zip(lines, weights)))
    left = min(line.x1 for line in lines)
    right = max(line.x2 for line in lines)
    score = float(sum(line.score for line in lines) + 0.20 * (right - left))
    return Line(
        left,
        anchor_y + slope * (left - anchor_x),
        right,
        anchor_y + slope * (right - anchor_x),
        score,
    )


def lines_can_merge(a: Line, b: Line, cfg: Config) -> bool:
    if horizontal_gap(a, b) > cfg.merge_x_gap_px:
        return False
    if abs(a.slope - b.slope) > cfg.merge_slope_delta:
        return False
    overlap_left = max(a.x1, b.x1)
    overlap_right = min(a.x2, b.x2)
    if overlap_right >= overlap_left:
        reference_x = (overlap_left + overlap_right) * 0.5
    elif a.x2 < b.x1:
        reference_x = (a.x2 + b.x1) * 0.5
    else:
        reference_x = (b.x2 + a.x1) * 0.5
    return abs(a.y_at(reference_x) - b.y_at(reference_x)) <= cfg.merge_y_px


def merge_similar_lines(lines: Sequence[Line], cfg: Config) -> List[Line]:
    clusters: List[List[Line]] = []
    for line in sorted(lines, key=lambda item: item.score, reverse=True):
        match_index: Optional[int] = None
        best_distance = float("inf")
        for index, cluster in enumerate(clusters):
            representative = merge_cluster(cluster)
            if not lines_can_merge(line, representative, cfg):
                continue
            distance = abs(line.center_y - representative.center_y)
            if distance < best_distance:
                match_index, best_distance = index, distance
        if match_index is None:
            clusters.append([line])
        else:
            clusters[match_index].append(line)

    merged = [merge_cluster(cluster) for cluster in clusters]
    merged = [
        line for line in merged if line.length_x >= cfg.min_boundary_length_px
    ]
    return sorted(merged, key=lambda item: item.score, reverse=True)[
        : cfg.max_merged_lines
    ]


def overlap_ratio(a: Line, b: Line) -> float:
    common = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    return common / max(min(a.length_x, b.length_x), 1.0)


def depth_patch_median(
    depth_m: np.ndarray, x: float, y: float, cfg: Config
) -> Optional[float]:
    radius = cfg.depth_patch_radius_px
    cx, cy = round(x), round(y)
    x0, x1 = max(0, cx - radius), min(depth_m.shape[1], cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(depth_m.shape[0], cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[
        np.isfinite(patch)
        & (patch >= cfg.depth_min_m)
        & (patch <= cfg.depth_max_m)
    ]
    return float(np.median(valid)) if valid.size >= 3 else None


def sample_z_columns(
    depth_m: np.ndarray,
    line: Line,
    left: float,
    right: float,
    side: int,
    offsets: Iterable[int],
    samples: int,
    cfg: Config,
) -> np.ndarray:
    values: List[float] = []
    for x in np.linspace(left, right, samples):
        depths = [
            depth_patch_median(depth_m, x, line.y_at(float(x)) + side * offset, cfg)
            for offset in offsets
        ]
        depths = [value for value in depths if value is not None]
        if depths:
            values.append(float(np.median(depths)))
    return np.asarray(values, np.float64)


def sample_face_layer_z(
    depth_m: np.ndarray,
    upper: Line,
    lower: Line,
    left: float,
    right: float,
    fraction: float,
    cfg: Config,
) -> np.ndarray:
    values: List[float] = []
    for x in np.linspace(left, right, cfg.evidence_x_samples):
        upper_y = upper.y_at(float(x))
        lower_y = lower.y_at(float(x))
        y = upper_y + fraction * (lower_y - upper_y)
        value = depth_patch_median(depth_m, x, y, cfg)
        if value is not None:
            values.append(value)
    return np.asarray(values, np.float64)


def assess_pair_depth(
    depth_m: np.ndarray, pair: BoundaryPair, cfg: Config
) -> PairEvidence:
    upper, lower = pair
    left = max(upper.x1, lower.x1) + cfg.height_x_inset_px
    right = min(upper.x2, lower.x2) - cfg.height_x_inset_px
    nan = float("nan")
    if right <= left:
        return PairEvidence(0, nan, nan, nan, nan, nan, nan, 0, 0, 0)

    top = sample_z_columns(
        depth_m, upper, left, right, -1, cfg.evidence_offsets_px,
        cfg.evidence_x_samples, cfg,
    )
    floor = sample_z_columns(
        depth_m, lower, left, right, +1, cfg.evidence_offsets_px,
        cfg.evidence_x_samples, cfg,
    )
    face_layers = [
        sample_face_layer_z(depth_m, upper, lower, left, right, fraction, cfg)
        for fraction in (0.18, 0.50, 0.82)
    ]
    face_medians = [
        float(np.median(values))
        for values in face_layers
        if values.size >= cfg.evidence_min_valid_columns
    ]
    top_z = float(np.median(top)) if top.size else nan
    floor_z = float(np.median(floor)) if floor.size else nan
    face_z = float(np.median(face_medians)) if face_medians else nan
    face_delta = (
        float(max(face_medians) - min(face_medians))
        if len(face_medians) == 3 else float("inf")
    )

    enough = (
        top.size >= cfg.evidence_min_valid_columns
        and floor.size >= cfg.evidence_min_valid_columns
        and len(face_medians) == 3
    )
    if enough:
        # Absolute transitions keep the same detector usable for an up- or
        # down-step. Height itself is still calculated only from 3D Y medians.
        upper_transition = abs(top_z - face_medians[0])
        lower_transition = abs(face_medians[-1] - floor_z)
        upper_ok = upper_transition >= cfg.evidence_min_transition_m
        lower_ok = lower_transition >= cfg.evidence_min_transition_m
        face_ok = face_delta <= cfg.evidence_max_face_layer_delta_m
        if upper_ok and lower_ok and face_ok:
            tier = 3
        elif upper_ok and lower_ok:
            tier = 2
        elif (upper_ok or lower_ok) and face_ok:
            tier = 1
        else:
            tier = 0
    else:
        upper_transition = lower_transition = nan
        tier = 0

    face_valid = min((values.size for values in face_layers), default=0)
    return PairEvidence(
        tier=tier,
        top_z_m=top_z,
        face_z_m=face_z,
        floor_z_m=floor_z,
        upper_transition_m=float(upper_transition),
        lower_transition_m=float(lower_transition),
        face_layer_delta_m=face_delta,
        top_valid=int(top.size),
        face_valid=int(face_valid),
        floor_valid=int(floor.size),
    )


def pair_geometry(
    first: Line, second: Line, cfg: Config
) -> Optional[Tuple[BoundaryPair, float]]:
    upper, lower = sorted((first, second), key=lambda line: line.center_y)
    left, right = max(upper.x1, lower.x1), min(upper.x2, lower.x2)
    common_width = right - left
    if common_width < cfg.min_common_width_px:
        return None
    if overlap_ratio(upper, lower) < cfg.min_overlap_ratio:
        return None
    if abs(upper.slope - lower.slope) > cfg.max_pair_slope_delta:
        return None

    center_x = (left + right) * 0.5
    gap = lower.y_at(center_x) - upper.y_at(center_x)
    if not cfg.min_boundary_gap_px <= gap <= cfg.max_boundary_gap_px:
        return None

    _, roi_y0, _, roi_y1 = roi_bounds(cfg)
    minimum_lower_y = roi_y0 + cfg.min_lower_position_in_roi * (roi_y1 - roi_y0)
    if lower.y_at(center_x) < minimum_lower_y:
        return None
    if roi_y1 - lower.y_at(center_x) < cfg.min_space_below_lower_px:
        return None

    upper = upper.crop(left, right)
    lower = lower.crop(left, right)
    length_similarity = min(first.length_x, second.length_x) / max(
        first.length_x, second.length_x, 1.0
    )
    slope_similarity = max(
        0.0, 1.0 - abs(upper.slope - lower.slope) / cfg.max_pair_slope_delta
    )
    rgb_score = (
        first.score + second.score
        + 1.8 * common_width
        + 120.0 * length_similarity
        + 90.0 * slope_similarity
    )
    return (upper, lower), float(rgb_score)


def detect_boundary_pair(
    color_bgr: np.ndarray, depth_m: np.ndarray, cfg: Config
) -> Tuple[Optional[BoundaryPair], Optional[PairEvidence]]:
    lines = merge_similar_lines(find_horizontal_lines(color_bgr, cfg), cfg)
    best_pair: Optional[BoundaryPair] = None
    best_evidence: Optional[PairEvidence] = None
    best_rank = (-1, -float("inf"))

    for index, first in enumerate(lines[:-1]):
        for second in lines[index + 1:]:
            geometry = pair_geometry(first, second, cfg)
            if geometry is None:
                continue
            pair, rgb_score = geometry
            evidence = assess_pair_depth(depth_m, pair, cfg)
            if evidence.tier < cfg.min_evidence_tier:
                continue
            transition_score = 0.0
            if np.isfinite(evidence.upper_transition_m):
                transition_score += min(evidence.upper_transition_m, 0.08) * 2500.0
            if np.isfinite(evidence.lower_transition_m):
                transition_score += min(evidence.lower_transition_m, 0.08) * 2500.0
            rank = (evidence.tier, rgb_score + transition_score)
            if rank > best_rank:
                best_pair, best_evidence, best_rank = pair, evidence, rank
    return best_pair, best_evidence


def pair_array(pair: BoundaryPair) -> np.ndarray:
    return np.stack((pair[0].as_array(), pair[1].as_array()))


def pair_from_array(values: np.ndarray) -> BoundaryPair:
    return Line.from_array(values[0]), Line.from_array(values[1])


def pair_is_close(a: np.ndarray, b: np.ndarray, cfg: Config) -> bool:
    pair_a = pair_from_array(a)
    pair_b = pair_from_array(b)

    # Hough 선분의 X 시작·끝 위치가 달라도
    # ROI 중앙에서 같은 경계선인지 비교
    roi_x0, _, roi_x1, _ = roi_bounds(cfg)
    reference_x = (roi_x0 + roi_x1) * 0.5

    a_y = np.asarray(
        [line.y_at(reference_x) for line in pair_a],
        dtype=np.float64,
    )
    b_y = np.asarray(
        [line.y_at(reference_x) for line in pair_b],
        dtype=np.float64,
    )

    a_mid_y = float(np.mean(a_y))
    b_mid_y = float(np.mean(b_y))

    a_gap = float(a_y[1] - a_y[0])
    b_gap = float(b_y[1] - b_y[0])

    slope_delta = max(
        abs(pair_a[0].slope - pair_b[0].slope),
        abs(pair_a[1].slope - pair_b[1].slope),
    )

    return (
        abs(a_mid_y - b_mid_y) <= cfg.boundary_y_tolerance_px
        and abs(a_gap - b_gap) <= cfg.boundary_gap_tolerance_px
        and slope_delta <= cfg.boundary_slope_tolerance
    )


class BoundaryTracker:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.history: Deque[Optional[np.ndarray]] = deque(
            maxlen=cfg.boundary_window_frames
        )
        self.confirmed: Optional[np.ndarray] = None
        self.missed = 0

    def reset(self) -> None:
        self.history.clear()
        self.confirmed = None
        self.missed = 0

    def update(self, pair: Optional[BoundaryPair]) -> Optional[BoundaryPair]:
        current = pair_array(pair) if pair is not None else None
        self.history.append(current)

        if current is None:
            self.missed += 1
            if self.missed > self.cfg.max_missed_frames:
                self.reset()
            return pair_from_array(self.confirmed) if self.confirmed is not None else None

        self.missed = 0
        consistent = [
            saved for saved in self.history
            if saved is not None and pair_is_close(current, saved, self.cfg)
        ]
        if len(consistent) >= self.cfg.boundary_required_frames:
            voted = np.median(np.stack(consistent), axis=0)
            if self.confirmed is None or not pair_is_close(voted, self.confirmed, self.cfg):
                self.confirmed = voted
            else:
                alpha = self.cfg.boundary_smoothing_alpha
                self.confirmed = (1.0 - alpha) * self.confirmed + alpha * voted
        return pair_from_array(self.confirmed) if self.confirmed is not None else None

    def current_matches(self, pair: Optional[BoundaryPair]) -> bool:
        return (
            pair is not None
            and self.confirmed is not None
            and pair_is_close(pair_array(pair), self.confirmed, self.cfg)
        )


def robust_surface_mask(values: np.ndarray, cfg: Config) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return finite
    center = float(np.median(values[finite]))
    spread = robust_mad(values[finite])
    tolerance = max(
        cfg.surface_outlier_min_tolerance_m,
        cfg.surface_outlier_mad_scale * spread,
    )
    return finite & (np.abs(values - center) <= tolerance)


def sample_surface_xyz(
    depth_m: np.ndarray,
    line: Line,
    xs: np.ndarray,
    side: int,
    deproject: Deproject,
    cfg: Config,
) -> Optional[SurfaceStats]:
    y_columns = np.full(xs.shape, np.nan, np.float64)
    z_columns = np.full(xs.shape, np.nan, np.float64)

    for index, x in enumerate(xs):
        y_values: List[float] = []
        z_values: List[float] = []
        boundary_y = line.y_at(float(x))
        for offset in cfg.height_band_offsets_px:
            pixel_y = boundary_y + side * offset
            depth = depth_patch_median(depth_m, float(x), pixel_y, cfg)
            if depth is None:
                continue
            point = np.asarray(
                deproject((float(x), float(pixel_y)), float(depth)), np.float64
            )
            if point.size < 3 or not np.all(np.isfinite(point[:3])):
                continue
            # D435 optical coordinates: +Y points downward. With the camera
            # mounted level, this is the vertical coordinate requested here.
            y_values.append(float(point[1]))
            z_values.append(float(point[2]))
        if len(y_values) >= 3:
            y_columns[index] = float(np.median(y_values))
            z_columns[index] = float(np.median(z_values))

    keep = robust_surface_mask(y_columns, cfg)
    if int(np.count_nonzero(keep)) < cfg.height_min_valid_columns:
        return None
    y_columns[~keep] = np.nan
    z_columns[~keep] = np.nan
    valid_y = y_columns[keep]
    valid_z = z_columns[keep]
    return SurfaceStats(
        median_y_m=float(np.median(valid_y)),
        median_z_m=float(np.median(valid_z)),
        spread_y_m=robust_mad(valid_y),
        valid_columns=int(valid_y.size),
        y_by_column=y_columns,
    )

def adaptive_height_offsets(
    front_z_m: float,
    cfg: Config,
) -> Tuple[int, ...]:
    """Z거리에 따라 경계선에서 떨어지는 픽셀 거리를 조절한다.

    약 70cm에서는 기존 6, 7, 8px를 그대로 사용하고,
    가까워질수록 더 안쪽을 측정한다.
    """
    if not np.isfinite(front_z_m):
        return cfg.height_band_offsets_px

    reference_z_m = 0.70

    scale = float(np.clip(
        reference_z_m / max(front_z_m, 1e-6),
        1.0,
        1.5,
    ))

    offsets = tuple(sorted(set(
        max(1, int(round(offset * scale)))
        for offset in cfg.height_band_offsets_px
    )))

    return offsets

def measure_height_from_two_boundaries(
    depth_m: np.ndarray,
    pair: Optional[BoundaryPair],
    deproject: Deproject,
    cfg: Config,
    evidence: Optional[PairEvidence] = None,
) -> Optional[HeightMeasurement]:
    """Return abs(median(Y_lower) - median(Y_upper)); nothing else."""
    if pair is None or depth_m.ndim != 2:
        return None
    upper, lower = pair
    left = max(upper.x1, lower.x1) + cfg.height_x_inset_px
    right = min(upper.x2, lower.x2) - cfg.height_x_inset_px
    if right <= left:
        return None

    xs = np.linspace(left, right, cfg.height_x_samples)
    upper_surface = sample_surface_xyz(
        depth_m, upper, xs, -1, deproject, cfg
    )
    lower_surface = sample_surface_xyz(
        depth_m, lower, xs, +1, deproject, cfg
    )
    if upper_surface is None or lower_surface is None:
        return None
    if (
        upper_surface.spread_y_m > cfg.surface_max_y_spread_m
        or lower_surface.spread_y_m > cfg.surface_max_y_spread_m
    ):
        return None

    # This is the requested height equation. No RGB pixel-gap conversion,
    # common face Z, plane fitting, or face-height approximation is used.
    signed_height = lower_surface.median_y_m - upper_surface.median_y_m
    raw_height = abs(signed_height)
    if not cfg.height_min_m <= raw_height <= cfg.height_max_m:
        return None

    paired_mask = (
        np.isfinite(upper_surface.y_by_column)
        & np.isfinite(lower_surface.y_by_column)
    )
    paired_heights = np.abs(
        lower_surface.y_by_column[paired_mask]
        - upper_surface.y_by_column[paired_mask]
    )
    if paired_heights.size < cfg.height_min_paired_columns:
        return None
    column_spread = robust_mad(paired_heights)
    if column_spread > cfg.height_max_column_spread_m:
        return None

    front_z = (
        evidence.face_z_m
        if evidence is not None and np.isfinite(evidence.face_z_m)
        else float(np.median((upper_surface.median_z_m, lower_surface.median_z_m)))
    )
    if not cfg.depth_min_m <= front_z <= cfg.depth_max_m:
        return None

    return HeightMeasurement(
        height_m=raw_height,
        raw_height_m=raw_height,
        signed_height_m=float(signed_height),
        upper_y_m=upper_surface.median_y_m,
        lower_y_m=lower_surface.median_y_m,
        front_z_m=float(front_z),
        upper_valid=upper_surface.valid_columns,
        lower_valid=lower_surface.valid_columns,
        paired_valid=int(paired_heights.size),
        total_columns=cfg.height_x_samples,
        column_spread_m=float(column_spread),
    )

class HeightTracker:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        self.history: Deque[float] = deque(
            maxlen=cfg.height_history_frames
        )
        self.z_history: Deque[float] = deque(
            maxlen=cfg.height_history_frames
        )

        self.last_output: Optional[HeightMeasurement] = None
        self.missed = 0

    def reset(self) -> None:
        self.history.clear()
        self.z_history.clear()
        self.last_output = None
        self.missed = 0

    @property
    def valid_frames(self) -> int:
        return len(self.history)

    def update(
        self,
        measurement: Optional[HeightMeasurement],
    ) -> Optional[HeightMeasurement]:

        # 현재 프레임이 일시적으로 실패한 경우
        if measurement is None:
            self.missed += 1

            if self.missed > self.cfg.max_missed_frames:
                self.reset()
                return None

            # 1~2프레임 정도의 순간적인 Depth 누락은
            # 직전 정상 측정값을 유지
            if (
                self.last_output is not None
                and self.missed <= self.cfg.height_hold_frames
            ):
                return self.last_output

            return None

        self.missed = 0

        # 기존 높이와 갑자기 4cm 이상 달라지면
        # 다른 물체 또는 오검출로 보고 이력 초기화
        if self.history:
            previous = float(np.median(self.history))

            if (
                abs(measurement.raw_height_m - previous)
                > self.cfg.height_history_reset_m
            ):
                self.history.clear()
                self.z_history.clear()
                self.last_output = None

        self.history.append(measurement.raw_height_m)
        self.z_history.append(measurement.front_z_m)

        if len(self.history) < self.cfg.height_required_frames:
            return None

        height_values = np.asarray(
            self.history,
            dtype=np.float64,
        )
        z_values = np.asarray(
            self.z_history,
            dtype=np.float64,
        )

        temporal_spread = robust_mad(height_values)

        if temporal_spread > self.cfg.height_max_temporal_spread_m:
            return None

        output = replace(
            measurement,

            # 높이 중앙값 안정화
            height_m=float(np.median(height_values)),

            # Z값도 동일한 프레임 이력의 중앙값으로 안정화
            front_z_m=float(np.median(z_values)),

            temporal_spread_m=float(temporal_spread),
        )

        self.last_output = output
        return output

def center_crop_line(line: Line, max_length: float) -> Line:
    if line.length_x <= max_length:
        return line
    center = line.center_x
    return line.crop(center - max_length * 0.5, center + max_length * 0.5)


def draw_result(
    color_bgr: np.ndarray,
    pair: Optional[BoundaryPair],
    raw_measurement: Optional[HeightMeasurement],
    stable_measurement: Optional[HeightMeasurement],
    height_tracker: HeightTracker,
    cfg: Config,
    show_roi: bool,
) -> np.ndarray:
    display = color_bgr.copy()
    if show_roi:
        x0, y0, x1, y1 = roi_bounds(cfg)
        cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)
        _outlined_text(display, "RGB DETECTION ROI", (x0 + 8, y0 + 24), 0.55, (0, 255, 255))

    if pair is None:
        _outlined_text(display, "SEARCHING: TWO STEP BOUNDARIES", (16, 34), 0.66, (255, 255, 255))
        return display

    for line, color in zip(pair, ((255, 255, 0), (255, 0, 255))):
        visible = center_crop_line(line, cfg.display_line_length_px)
        p1 = (round(visible.x1), round(visible.y1))
        p2 = (round(visible.x2), round(visible.y2))
        cv2.line(display, p1, p2, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.line(display, p1, p2, color, 4, cv2.LINE_AA)

    if stable_measurement is not None:
        item = stable_measurement
        _outlined_text(
            display,
            f"STEP HEIGHT: {item.height_m * 100:.1f} cm",
            (16, 34), 0.72, (0, 255, 0),
        )
        _outlined_text(
            display,
            f"Z: {item.front_z_m * 100:.1f} cm | VALID U/L: "
            f"{item.upper_valid}/{item.lower_valid} | SPREAD: "
            f"{item.column_spread_m * 100:.2f} cm",
            (16, 62), 0.52, (0, 255, 0),
        )
    elif raw_measurement is not None:
        _outlined_text(
            display,
            f"STABILIZING HEIGHT: {height_tracker.valid_frames}/"
            f"{cfg.height_required_frames}",
            (16, 34), 0.66, (0, 255, 255),
        )
        _outlined_text(
            display,
            f"RAW: {raw_measurement.raw_height_m * 100:.1f} cm | VALID U/L: "
            f"{raw_measurement.upper_valid}/{raw_measurement.lower_valid}",
            (16, 62), 0.52, (0, 255, 255),
        )
    else:
        _outlined_text(
            display, "TWO BOUNDARIES | DEPTH/HEIGHT NOT VALID",
            (16, 34), 0.62, (0, 255, 255),
        )
        _outlined_text(
            display,
            "Need Z 30-100 cm and enough samples in both 8-10 px bands",
            (16, 62), 0.46, (0, 255, 255),
        )
    return display


def configure_depth_filters(rs: object) -> Tuple[object, object]:
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    try:
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.50)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        temporal.set_option(rs.option.filter_smooth_alpha, 0.45)
        temporal.set_option(rs.option.filter_smooth_delta, 20)
    except Exception:
        # Defaults are still safe when an older pyrealsense2 build does not
        # expose one of the optional settings.
        pass
    return spatial, temporal


def run_live() -> None:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is required. Run this where realsense-viewer works."
        ) from exc

    cfg = Config()
    boundary_tracker = BoundaryTracker(cfg)
    height_tracker = HeightTracker(cfg)
    show_roi = True

    pipeline = rs.pipeline()
    stream_config = rs.config()
    stream_config.enable_stream(
        rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps
    )
    stream_config.enable_stream(
        rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps
    )
    profile = pipeline.start(stream_config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align_to_color = rs.align(rs.stream.color)
    spatial, temporal = configure_depth_filters(rs)

    print(f"Build: {BUILD}")
    print(f"D435: {cfg.width}x{cfg.height}@{cfg.fps} FPS")
    print("Height = abs(median(Y_lower) - median(Y_upper))")
    print("q/ESC: quit, r: reset, i: show/hide ROI")

    try:
        while True:
            aligned = align_to_color.process(pipeline.wait_for_frames())
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            if cfg.use_realsense_filters:
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_frame = depth_frame.as_depth_frame()

            color = np.asanyarray(color_frame.get_data())
            depth_m = (
                np.asanyarray(depth_frame.get_data()).astype(np.float32)
                * depth_scale
            )
            intrinsics = (
                depth_frame.profile.as_video_stream_profile().get_intrinsics()
            )

            def deproject(
                pixel: Tuple[float, float], depth: float
            ) -> Sequence[float]:
                return rs.rs2_deproject_pixel_to_point(
                    intrinsics, [float(pixel[0]), float(pixel[1])], float(depth)
                )

            detected, detected_evidence = detect_boundary_pair(
                color, depth_m, cfg
            )

            confirmed = boundary_tracker.update(detected)

            raw_measurement = None

            if (
                detected is not None
                and confirmed is not None
                and boundary_tracker.current_matches(detected)
                and detected_evidence is not None
                and detected_evidence.tier >= cfg.min_evidence_tier
            ):
                measure_pair_array = (
                0.8 * pair_array(confirmed)
                + 0.2 * pair_array(detected)
                )
                measure_pair = pair_from_array(measure_pair_array)
                raw_measurement = measure_height_from_two_boundaries(
            depth_m,
            measure_pair,
            deproject,
            cfg,
            detected_evidence,  # 검출 당시 통과한 evidence 사용
            )
            stable_measurement = height_tracker.update(raw_measurement)

            display = draw_result(
                color,
                confirmed,
                raw_measurement,
                stable_measurement,
                height_tracker,
                cfg,
                show_roi,
            )
            cv2.imshow("D435 - Two Boundaries Median-Y Height", display)
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