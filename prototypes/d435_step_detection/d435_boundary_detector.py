#!/usr/bin/env python3
"""Detect two step boundaries on the Intel RealSense D435 RGB stream.

The detector follows the same pipeline used for the recorded RGB height test:

    RGB -> Gray -> CLAHE -> Gaussian blur -> Canny -> HoughLinesP

Canny images and Hough candidates are used only internally. The one output
window always shows the real RGB image and, after temporal confirmation, only
the final upper/lower boundary pair.

Controls
--------
q or ESC : quit
r        : reset temporal tracking
i        : show/hide the RGB detection ROI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


DETECTOR_BUILD = "RGB848_PARTIAL_V8"


@dataclass(frozen=True)
class DetectorConfig:
    # Final Windows/robot test profile. D435 supports 848x480 RGB at 30 FPS.
    width: int = 848
    height: int = 480
    fps: int = 30

    # Temporary lower-central bench-test ROI. Recalibrate these four ratios
    # after the D435 is rigidly mounted on the robot.
    roi_left: float = 0.20
    roi_top: float = 0.35
    roi_right: float = 0.80
    roi_bottom: float = 1.00

    # Settings preserved from the successful recorded-RGB verification.
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    gaussian_kernel: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.0
    canny_low: int = 50
    canny_high: int = 150
    # Slightly relaxed from the recorded-video values (50/50/10) because live
    # auto-exposure breaks the same physical edge into shorter pieces.
    hough_threshold: int = 40
    hough_min_line_length_px: int = 40
    hough_max_line_gap_px: int = 20

    # Horizontal filtering and duplicate-line merging.
    max_angle_deg: float = 7.0
    merge_y_px: float = 9.0
    # A useful boundary position does not require the full physical edge.
    # Keep partial edges, while Hough's own minimum rejects tiny fragments.
    min_final_length_ratio: float = 0.08

    # A valid step must produce a separated upper/lower line pair.
    min_boundary_gap_px: float = 18.0
    # Earlier RGB tests produced about 60-111 px for the 14.5 cm box at
    # 100-50 cm. 160 px keeps margin without pairing distant room edges.
    max_boundary_gap_px: float = 160.0
    min_pair_overlap_ratio: float = 0.35
    # The two boundaries need only share enough width to identify their Y
    # positions. At the default ROI this is about 127 px, not the full edge.
    min_pair_common_width_ratio_of_roi: float = 0.25
    min_lower_boundary_y_ratio: float = 0.50

    # A true lower-front boundary must have a visible floor region below it.
    # Strong vertical texture continuing below the line indicates that the
    # selected line is still on the box/top/front face, not the floor contact.
    floor_band_start_px: int = 8
    floor_band_end_px: int = 40
    # Near 50 cm the real floor-contact edge can approach the ROI bottom.
    # Accept a clipped floor band when at least this many pixels remain.
    floor_min_visible_band_px: int = 16
    floor_side_gap_px: int = 8
    floor_min_side_width_px: int = 16
    floor_vertical_gradient_threshold: float = 30.0
    floor_max_edge_density: float = 0.075
    floor_max_density_over_reference: float = 0.060

    # Show a pair only after it stays at nearly the same Y position for five
    # consecutive frames. This suppresses one-frame clutter selections.
    confirm_frames: int = 5
    tracking_tolerance_px: float = 14.0
    smoothing_alpha: float = 0.25
    max_missed_frames: int = 12

    # Drawing is intentionally shorter than the detected support. The full
    # BoundaryLine remains available for later depth-plane sampling.
    display_line_length_px: float = 160.0


@dataclass
class LineCandidate:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    contrast: float
    score: float

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def slope(self) -> float:
        dx = self.x2 - self.x1
        return 0.0 if abs(dx) < 1e-6 else (self.y2 - self.y1) / dx


@dataclass
class BoundaryLine:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def length_x(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @staticmethod
    def from_array(values: np.ndarray, score: float = 0.0) -> "BoundaryLine":
        return BoundaryLine(*(float(v) for v in values), score=float(score))


BoundaryPair = Tuple[BoundaryLine, BoundaryLine]


@dataclass(frozen=True)
class DetectionResult:
    pair: Optional[BoundaryPair]
    status: str


STATUS_SEARCHING = "SEARCHING FOR TWO RGB BOUNDARIES"
STATUS_TOO_CLOSE = "LOWER EDGE NOT VISIBLE / TOO CLOSE"
STATUS_DETECTED = "RGB BOUNDARIES DETECTED"


def roi_pixels(cfg: DetectorConfig) -> Tuple[int, int, int, int]:
    return (
        int(round(cfg.width * cfg.roi_left)),
        int(round(cfg.height * cfg.roi_top)),
        int(round(cfg.width * cfg.roi_right)),
        int(round(cfg.height * cfg.roi_bottom)),
    )


def build_hidden_edge_image(
    color_bgr: np.ndarray, cfg: DetectorConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the internal Canny image and enhanced gray image."""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_size,
    )
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(
        enhanced,
        cfg.gaussian_kernel,
        cfg.gaussian_sigma,
    )
    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

    x0, y0, x1, y1 = roi_pixels(cfg)
    roi_mask = np.zeros_like(edges)
    roi_mask[y0:y1, x0:x1] = 255
    edges = cv2.bitwise_and(edges, roi_mask)
    return edges, enhanced


def contrast_across_line(
    gray: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Return median RGB-intensity change above and below one line."""
    left = int(round(min(x1, x2)))
    right = int(round(max(x1, x2)))
    if right - left < 8:
        return 0.0

    sample_count = min(160, right - left + 1)
    xs = np.linspace(left, right, sample_count).astype(np.int32)
    slope = (y2 - y1) / max(x2 - x1, 1e-6)
    ys = np.rint(y1 + slope * (xs - x1)).astype(np.int32)

    upper_values: List[np.ndarray] = []
    lower_values: List[np.ndarray] = []
    for offset in (3, 4, 5):
        yu = ys - offset
        yl = ys + offset
        inside_u = (yu >= 0) & (yu < gray.shape[0])
        inside_l = (yl >= 0) & (yl < gray.shape[0])
        if np.any(inside_u):
            upper_values.append(gray[yu[inside_u], xs[inside_u]])
        if np.any(inside_l):
            lower_values.append(gray[yl[inside_l], xs[inside_l]])

    if not upper_values or not lower_values:
        return 0.0
    upper = np.concatenate(upper_values)
    lower = np.concatenate(lower_values)
    return abs(float(np.median(upper)) - float(np.median(lower)))


def find_horizontal_candidates(
    color_bgr: np.ndarray,
    cfg: DetectorConfig,
) -> List[LineCandidate]:
    edges, enhanced_gray = build_hidden_edge_image(color_bgr, cfg)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_length_px,
        maxLineGap=cfg.hough_max_line_gap_px,
    )
    if raw_lines is None:
        return []

    x0, _, x1_roi, _ = roi_pixels(cfg)
    roi_center_x = 0.5 * (x0 + x1_roi)
    roi_width = max(1.0, float(x1_roi - x0))
    candidates: List[LineCandidate] = []

    # OpenCV builds return either (N, 1, 4) or (N, 4).
    for packed in np.asarray(raw_lines).reshape(-1, 4):
        x1, y1, x2, y2 = (float(v) for v in packed)
        if x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        angle_deg = abs(float(np.degrees(np.arctan2(dy, dx))))
        # Keep every segment that already passed Hough's minimum. Short
        # pieces at the same Y are merged before the final-length test.
        if angle_deg > cfg.max_angle_deg:
            continue

        contrast = contrast_across_line(enhanced_gray, x1, y1, x2, y2)
        angle_factor = max(0.0, 1.0 - angle_deg / cfg.max_angle_deg)
        center_distance = abs(0.5 * (x1 + x2) - roi_center_x) / roi_width
        center_factor = max(0.35, 1.0 - center_distance)
        contrast_factor = 1.0 + min(contrast / 50.0, 1.0) * 0.30
        score = length * (0.60 + 0.40 * angle_factor)
        score *= center_factor * contrast_factor

        candidates.append(
            LineCandidate(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                length=length,
                angle_deg=angle_deg,
                contrast=contrast,
                score=score,
            )
        )
    return candidates


def merge_similar_lines(
    candidates: Sequence[LineCandidate],
    cfg: DetectorConfig,
) -> List[BoundaryLine]:
    """Merge Hough segments at similar Y positions into clean boundaries."""
    if not candidates:
        return []

    clusters: List[List[LineCandidate]] = []
    for candidate in sorted(candidates, key=lambda line: line.center_y):
        selected_cluster: Optional[List[LineCandidate]] = None
        selected_distance = float("inf")
        for cluster in clusters:
            weights = [line.score for line in cluster]
            cluster_y = float(
                np.average([line.center_y for line in cluster], weights=weights)
            )
            distance = abs(candidate.center_y - cluster_y)
            if distance <= cfg.merge_y_px and distance < selected_distance:
                selected_cluster = cluster
                selected_distance = distance

        if selected_cluster is None:
            clusters.append([candidate])
        else:
            selected_cluster.append(candidate)

    merged: List[BoundaryLine] = []
    min_final_length = cfg.width * cfg.min_final_length_ratio
    for cluster in clusters:
        weights = np.array([line.score for line in cluster], dtype=np.float64)
        weights /= max(float(weights.sum()), 1e-9)
        center_y = float(
            np.sum([line.center_y * w for line, w in zip(cluster, weights)])
        )
        slope = float(
            np.sum([line.slope * w for line, w in zip(cluster, weights)])
        )
        x1 = float(min(line.x1 for line in cluster))
        x2 = float(max(line.x2 for line in cluster))
        if x2 - x1 < min_final_length:
            continue

        center_x = 0.5 * (x1 + x2)
        y1 = center_y + slope * (x1 - center_x)
        y2 = center_y + slope * (x2 - center_x)
        coverage_bonus = 0.25 * (x2 - x1)
        score = float(sum(line.score for line in cluster)) + coverage_bonus
        merged.append(BoundaryLine(x1, y1, x2, y2, score))
    return merged


def horizontal_overlap_ratio(a: BoundaryLine, b: BoundaryLine) -> float:
    overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    shorter = max(1.0, min(a.length_x, b.length_x))
    return overlap / shorter


def crop_line_to_x_range(
    line: BoundaryLine,
    left: float,
    right: float,
) -> BoundaryLine:
    dx = max(line.x2 - line.x1, 1e-6)
    slope = (line.y2 - line.y1) / dx
    y_left = line.y1 + slope * (left - line.x1)
    y_right = line.y1 + slope * (right - line.x1)
    return BoundaryLine(left, y_left, right, y_right, line.score)


def center_crop_line(
    line: BoundaryLine,
    max_length_px: Optional[float],
) -> BoundaryLine:
    """Return only the central display segment without changing detection."""
    if max_length_px is None or max_length_px <= 0 or line.length_x <= max_length_px:
        return line
    center_x = 0.5 * (line.x1 + line.x2)
    half_length = 0.5 * max_length_px
    return crop_line_to_x_range(line, center_x - half_length, center_x + half_length)


def make_vertical_gradient_image(color_bgr: np.ndarray) -> np.ndarray:
    """Return X-gradient magnitude; strong values represent vertical texture."""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
    return np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))


def region_edge_density(
    vertical_gradient: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    threshold: float,
) -> Optional[float]:
    height, width = vertical_gradient.shape
    x0 = int(np.clip(x0, 0, width))
    x1 = int(np.clip(x1, 0, width))
    y0 = int(np.clip(y0, 0, height))
    y1 = int(np.clip(y1, 0, height))
    if x1 <= x0 or y1 <= y0:
        return None
    region = vertical_gradient[y0:y1, x0:x1]
    return float(np.mean(region >= threshold))


def floor_visible_below_pair(
    vertical_gradient: np.ndarray,
    pair: BoundaryPair,
    cfg: DetectorConfig,
) -> bool:
    """Verify that the lower line is followed by a visible floor-like band."""
    _, lower = pair
    roi_x0, _, roi_x1, roi_y1 = roi_pixels(cfg)

    line_y = int(round(lower.center_y))
    floor_y0 = line_y + cfg.floor_band_start_px
    floor_y1 = min(
        line_y + cfg.floor_band_end_px,
        roi_y1,
        vertical_gradient.shape[0],
    )
    if floor_y1 - floor_y0 < cfg.floor_min_visible_band_px:
        return False

    # Trim endpoints, where box side edges and shadows are expected.
    trim = max(4, int(round(lower.length_x * 0.08)))
    center_x0 = int(round(lower.x1)) + trim
    center_x1 = int(round(lower.x2)) - trim
    center_density = region_edge_density(
        vertical_gradient,
        center_x0,
        floor_y0,
        center_x1,
        floor_y1,
        cfg.floor_vertical_gradient_threshold,
    )
    if center_density is None:
        return False

    # Use the less-textured usable side as the local floor reference.
    side_densities: List[float] = []
    left_x1 = int(round(lower.x1)) - cfg.floor_side_gap_px
    if left_x1 - roi_x0 >= cfg.floor_min_side_width_px:
        value = region_edge_density(
            vertical_gradient,
            roi_x0,
            floor_y0,
            left_x1,
            floor_y1,
            cfg.floor_vertical_gradient_threshold,
        )
        if value is not None:
            side_densities.append(value)

    right_x0 = int(round(lower.x2)) + cfg.floor_side_gap_px
    if roi_x1 - right_x0 >= cfg.floor_min_side_width_px:
        value = region_edge_density(
            vertical_gradient,
            right_x0,
            floor_y0,
            roi_x1,
            floor_y1,
            cfg.floor_vertical_gradient_threshold,
        )
        if value is not None:
            side_densities.append(value)

    if center_density > cfg.floor_max_edge_density:
        return False
    if side_densities:
        reference_density = min(side_densities)
        if center_density > reference_density + cfg.floor_max_density_over_reference:
            return False
    return True


def select_final_pair(
    lines: Sequence[BoundaryLine],
    cfg: DetectorConfig,
    vertical_gradient: Optional[np.ndarray] = None,
) -> Tuple[Optional[BoundaryPair], bool]:
    """Select one upper/lower pair and discard every other Hough line."""
    if len(lines) < 2:
        return None, False

    best_pair: Optional[BoundaryPair] = None
    best_score = -float("inf")
    rejected_for_floor = False
    roi_center_x = 0.5 * cfg.width * (cfg.roi_left + cfg.roi_right)

    for i in range(len(lines) - 1):
        for j in range(i + 1, len(lines)):
            upper, lower = sorted((lines[i], lines[j]), key=lambda line: line.center_y)
            pixel_gap = lower.center_y - upper.center_y
            if not (cfg.min_boundary_gap_px <= pixel_gap <= cfg.max_boundary_gap_px):
                continue
            if lower.center_y < cfg.height * cfg.min_lower_boundary_y_ratio:
                continue

            overlap = horizontal_overlap_ratio(upper, lower)
            if overlap < cfg.min_pair_overlap_ratio:
                continue

            length_similarity = min(upper.length_x, lower.length_x) / max(
                upper.length_x,
                lower.length_x,
                1.0,
            )
            common_left = max(upper.x1, lower.x1)
            common_right = min(upper.x2, lower.x2)
            roi_width = cfg.width * (cfg.roi_right - cfg.roi_left)
            if (
                common_right - common_left
                < roi_width * cfg.min_pair_common_width_ratio_of_roi
            ):
                continue
            cropped_pair = (
                crop_line_to_x_range(upper, common_left, common_right),
                crop_line_to_x_range(lower, common_left, common_right),
            )
            # Validate the entire detected lower edge, not only the common
            # overlap. A short upper fragment can otherwise make a box face
            # look like a small, locally uniform patch of floor.
            if vertical_gradient is not None and not floor_visible_below_pair(
                vertical_gradient,
                (upper, lower),
                cfg,
            ):
                rejected_for_floor = True
                continue

            pair_center_x = 0.5 * (common_left + common_right)
            center_factor = max(
                0.0,
                1.0 - abs(pair_center_x - roi_center_x) / (0.5 * cfg.width),
            )

            pair_score = upper.score + lower.score
            pair_score += 180.0 * overlap
            pair_score += 100.0 * length_similarity
            pair_score += 70.0 * center_factor
            # The front-face top/bottom pair has a much larger vertical gap
            # than duplicate edge responses, box-top edges, or printed text.
            pair_score += 5.0 * pixel_gap
            if pair_score > best_score:
                best_pair = cropped_pair
                best_score = pair_score

    if best_pair is None:
        return None, rejected_for_floor
    return best_pair, rejected_for_floor


def detect_boundary_result(
    color_bgr: np.ndarray,
    cfg: DetectorConfig,
) -> DetectionResult:
    candidates = find_horizontal_candidates(color_bgr, cfg)
    merged = merge_similar_lines(candidates, cfg)
    vertical_gradient = make_vertical_gradient_image(color_bgr)
    pair, rejected_for_floor = select_final_pair(
        merged,
        cfg,
        vertical_gradient,
    )
    if pair is not None:
        return DetectionResult(pair, STATUS_DETECTED)
    if rejected_for_floor:
        return DetectionResult(None, STATUS_TOO_CLOSE)
    return DetectionResult(None, STATUS_SEARCHING)


def detect_boundary_pair(
    color_bgr: np.ndarray,
    cfg: DetectorConfig,
) -> Optional[BoundaryPair]:
    """Compatibility wrapper returning only the validated pair."""
    return detect_boundary_result(color_bgr, cfg).pair


class BoundaryTracker:
    """Confirm a pair and keep it through short RGB edge dropouts."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.pending: Optional[np.ndarray] = None
        self.confirmed: Optional[np.ndarray] = None
        self.consecutive = 0
        self.missed = 0

    def reset(self) -> None:
        self.pending = None
        self.confirmed = None
        self.consecutive = 0
        self.missed = 0

    @staticmethod
    def _pair_array(pair: BoundaryPair) -> np.ndarray:
        return np.stack([pair[0].as_array(), pair[1].as_array()])

    @staticmethod
    def _center_ys(values: np.ndarray) -> np.ndarray:
        return 0.5 * (values[:, 1] + values[:, 3])

    def update(self, pair: Optional[BoundaryPair]) -> Optional[BoundaryPair]:
        if pair is None:
            self.missed += 1
            self.pending = None
            self.consecutive = 0
            if self.missed > self.cfg.max_missed_frames:
                self.reset()
            return self._confirmed_pair()

        current = self._pair_array(pair)

        # If the new pair agrees with the displayed pair, update it directly.
        if self.confirmed is not None:
            confirmed_error = float(
                np.max(
                    np.abs(
                        self._center_ys(current) - self._center_ys(self.confirmed)
                    )
                )
            )
            if confirmed_error <= self.cfg.tracking_tolerance_px:
                alpha = self.cfg.smoothing_alpha
                self.confirmed = (1.0 - alpha) * self.confirmed + alpha * current
                self.pending = self.confirmed.copy()
                self.consecutive = self.cfg.confirm_frames
                self.missed = 0
                return self._confirmed_pair()

            # One inconsistent frame must not erase a previously stable pair.
            self.missed += 1
        else:
            self.missed = 0

        if self.pending is None:
            self.pending = current
            self.consecutive = 1
        else:
            pending_error = float(
                np.max(
                    np.abs(
                        self._center_ys(current) - self._center_ys(self.pending)
                    )
                )
            )
            if pending_error <= self.cfg.tracking_tolerance_px:
                alpha = self.cfg.smoothing_alpha
                self.pending = (1.0 - alpha) * self.pending + alpha * current
                self.consecutive += 1
            else:
                self.pending = current
                self.consecutive = 1

        if self.consecutive >= self.cfg.confirm_frames:
            self.confirmed = self.pending.copy()
            self.missed = 0
        elif self.missed > self.cfg.max_missed_frames:
            self.reset()
        return self._confirmed_pair()

    def _confirmed_pair(self) -> Optional[BoundaryPair]:
        if self.confirmed is None:
            return None
        return (
            BoundaryLine.from_array(self.confirmed[0]),
            BoundaryLine.from_array(self.confirmed[1]),
        )


def pair_pixel_gap(pair: BoundaryPair) -> float:
    return pair[1].center_y - pair[0].center_y


def draw_final_result(
    color_bgr: np.ndarray,
    pair: Optional[BoundaryPair],
    status: str = STATUS_SEARCHING,
    max_line_length_px: Optional[float] = None,
) -> np.ndarray:
    """Draw only central portions of the confirmed pair on the RGB frame."""
    display = color_bgr.copy()
    if pair is None:
        text = status
        text_color = (0, 200, 255) if status == STATUS_TOO_CLOSE else (255, 255, 255)
    else:
        # Upper: cyan, lower: magenta, as in the RGB verification display.
        line_colors = ((255, 255, 0), (255, 0, 255))
        for line, color in zip(pair, line_colors):
            visible_line = center_crop_line(line, max_line_length_px)
            p1 = (int(round(visible_line.x1)), int(round(visible_line.y1)))
            p2 = (int(round(visible_line.x2)), int(round(visible_line.y2)))
            cv2.line(display, p1, p2, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.line(display, p1, p2, color, 4, cv2.LINE_AA)
        text = f"RGB BOUNDARIES DETECTED | PIXEL GAP: {pair_pixel_gap(pair):.1f} px"
        text_color = (0, 255, 0)

    cv2.putText(
        display,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        text_color,
        2,
        cv2.LINE_AA,
    )
    return display


def draw_roi_overlay(color_bgr: np.ndarray, cfg: DetectorConfig) -> None:
    """Draw only the active RGB detection ROI on the final display."""
    x0, y0, x1, y1 = roi_pixels(cfg)
    color = (0, 255, 255)
    cv2.rectangle(
        color_bgr,
        (x0, y0),
        (x1 - 1, y1 - 1),
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        color_bgr,
        "RGB DETECTION ROI",
        (x0 + 8, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        color_bgr,
        "RGB DETECTION ROI",
        (x0 + 8, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def run_live() -> None:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is required. Run this file in the same environment "
            "where realsense-viewer works."
        ) from exc

    cfg = DetectorConfig()
    tracker = BoundaryTracker(cfg)
    show_roi = True
    pipeline = rs.pipeline()
    stream_cfg = rs.config()
    stream_cfg.enable_stream(
        rs.stream.color,
        cfg.width,
        cfg.height,
        rs.format.bgr8,
        cfg.fps,
    )

    pipeline.start(stream_cfg)
    print(f"Detector build: {DETECTOR_BUILD}")
    print(f"D435 RGB boundary detector: {cfg.width}x{cfg.height}@{cfg.fps} FPS")
    print("q/ESC: quit, r: reset tracking, i: show/hide ROI")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            result = detect_boundary_result(color_bgr, cfg)
            if result.status == STATUS_TOO_CLOSE:
                tracker.reset()
                confirmed = None
            else:
                confirmed = tracker.update(result.pair)

            display_status = STATUS_DETECTED if confirmed is not None else result.status
            display = draw_final_result(
                color_bgr,
                confirmed,
                display_status,
                cfg.display_line_length_px,
            )
            if show_roi:
                draw_roi_overlay(display, cfg)
            cv2.imshow("D435 RGB - Final Two Boundaries", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                tracker.reset()
            if key == ord("i"):
                show_roi = not show_roi
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live()
