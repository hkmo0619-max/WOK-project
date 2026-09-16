#!/usr/bin/env python3
"""Shared measurement engine for Intel RealSense D435 step-height tools.

The public entry points are ``automatic_step_height.py`` and
``manual_auto_step_height.py``.  This module keeps camera acquisition,
automatic edge selection, RANSAC floor fitting, robust height estimation,
temporal stabilization, drawing, and CSV logging in one place.

Coordinate convention follows librealsense: X right, Y down, Z forward.
All internal distances are metres.
"""

from __future__ import annotations

import csv
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # Allows plane-math tests on a machine without OpenCV.
    cv2 = None

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError:  # Allows plane-math tests without a connected D435 SDK.
    rs = None


ROI = Tuple[int, int, int, int]  # x, y, width, height


@dataclass(frozen=True)
class CameraConfig:
    depth_width: int = 848
    depth_height: int = 480
    color_width: int = 640
    color_height: int = 480
    fps: int = 30
    warmup_frames: int = 30
    min_depth_m: float = 0.25
    max_depth_m: float = 2.00
    use_filters: bool = True
    emitter_enabled: bool = True


@dataclass(frozen=True)
class DetectorConfig:
    # Broad automatic search region. This is not a manually selected ROI.
    search_x_min_ratio: float = 0.08
    search_x_max_ratio: float = 0.92
    search_y_min_ratio: float = 0.12
    search_y_max_ratio: float = 0.88

    # The foreground floor used for the reference plane.  Avoid the extreme
    # bottom of the depth-aligned image: D435 colour alignment can leave a
    # black border there, and the nearest floor can also be below min_depth_m.
    floor_x_min_ratio: float = 0.08
    floor_x_max_ratio: float = 0.92
    floor_y_min_ratio: float = 0.66
    floor_y_max_ratio: float = 0.90

    max_line_angle_deg: float = 8.0
    min_line_length_ratio: float = 0.22
    hough_threshold: int = 45
    hough_max_gap_px: int = 25
    max_line_candidates: int = 12

    # Bands are measured away from the detected top-front edge.
    edge_exclusion_px: int = 12
    top_band_height_px: int = 35
    face_band_height_px: int = 55
    horizontal_margin_px: int = 10

    min_step_height_m: float = 0.015
    max_step_height_m: float = 0.25
    min_floor_points: int = 600
    min_top_points: int = 150
    min_face_points: int = 60
    max_floor_points: int = 5000
    max_top_points: int = 4000

    ransac_iterations: int = 120
    ransac_threshold_m: float = 0.006
    min_floor_inlier_ratio: float = 0.55

    height_histogram_bin_m: float = 0.004
    height_cluster_radius_m: float = 0.012
    min_top_cluster_ratio: float = 0.35
    max_top_mad_m: float = 0.008
    min_candidate_confidence: float = 0.45

    depth_edge_threshold_m: float = 0.015
    camera_forward_offset_m: float = 0.0
    random_seed: int = 20260801


@dataclass(frozen=True)
class StabilityConfig:
    window_frames: int = 30
    required_frames: int = 15
    max_height_spread_m: float = 0.010
    max_edge_spread_px: float = 14.0
    max_tracking_height_jump_m: float = 0.040
    max_tracking_edge_jump_px: float = 30.0
    reset_after_misses: int = 10
    min_confidence: float = 0.45


@dataclass(frozen=True)
class PlaneModel:
    normal: np.ndarray
    d: float
    inlier_ratio: float
    rms_m: float
    inlier_count: int

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.abs(points @ self.normal + self.d)


@dataclass(frozen=True)
class PointSet:
    points: np.ndarray
    pixels: np.ndarray
    valid_ratio: float


@dataclass(frozen=True)
class Measurement:
    height_m: float
    distance_m: float
    center_offset_m: float
    confidence: float
    edge_y_px: float
    line: Optional[Tuple[int, int, int, int]]
    top_roi: ROI
    floor_roi: ROI
    face_roi: Optional[ROI]
    floor_inlier_ratio: float
    floor_rms_m: float
    top_mad_m: float
    top_cluster_ratio: float
    source: str


@dataclass(frozen=True)
class StableSummary:
    stable: bool
    samples: int
    height_m: float
    distance_m: float
    center_offset_m: float
    height_spread_m: float
    edge_spread_px: float
    confidence: float


@dataclass
class DetectionDebug:
    edge_map: Optional[np.ndarray] = None
    floor_roi: Optional[ROI] = None
    floor_plane: Optional[PlaneModel] = None
    floor_valid_points: int = 0
    floor_valid_ratio: float = 0.0
    candidate_count: int = 0
    reason: str = ""


@dataclass(frozen=True)
class SimpleIntrinsics:
    """Small test-friendly equivalent of librealsense intrinsics fields."""

    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int


def require_runtime_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if rs is None:
        missing.append("pyrealsense2")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"필수 패키지가 없습니다: {joined}. README의 설치 명령을 먼저 실행하세요."
        )


def clip_roi(roi: ROI, width: int, height: int) -> ROI:
    x, y, w, h = roi
    x1 = int(np.clip(x, 0, width))
    y1 = int(np.clip(y, 0, height))
    x2 = int(np.clip(x + w, 0, width))
    y2 = int(np.clip(y + h, 0, height))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def ratio_roi(
    width: int,
    height: int,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> ROI:
    return clip_roi(
        (
            int(round(width * x_min)),
            int(round(height * y_min)),
            int(round(width * (x_max - x_min))),
            int(round(height * (y_max - y_min))),
        ),
        width,
        height,
    )


def points_from_roi(
    depth_m: np.ndarray,
    intrinsics: Any,
    roi: ROI,
    min_depth_m: float,
    max_depth_m: float,
    max_points: int,
) -> PointSet:
    """Vectorized pixel deprojection for valid depth pixels inside one ROI."""

    image_h, image_w = depth_m.shape[:2]
    x, y, w, h = clip_roi(roi, image_w, image_h)
    if w <= 0 or h <= 0:
        return PointSet(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
            0.0,
        )

    crop = depth_m[y : y + h, x : x + w]
    valid = (
        np.isfinite(crop)
        & (crop >= min_depth_m)
        & (crop <= max_depth_m)
    )
    local_v, local_u = np.nonzero(valid)
    valid_count = int(local_u.size)
    valid_ratio = valid_count / float(w * h)
    if valid_count == 0:
        return PointSet(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
            valid_ratio,
        )

    u = local_u.astype(np.float64) + x
    v = local_v.astype(np.float64) + y
    z = crop[local_v, local_u].astype(np.float64)

    if max_points > 0 and valid_count > max_points:
        # Evenly spaced sampling keeps coverage across the full rectangle.
        keep = np.linspace(0, valid_count - 1, max_points, dtype=np.int64)
        u = u[keep]
        v = v[keep]
        z = z[keep]

    x3 = (u - float(intrinsics.ppx)) * z / float(intrinsics.fx)
    y3 = (v - float(intrinsics.ppy)) * z / float(intrinsics.fy)
    points = np.column_stack((x3, y3, z))
    pixels = np.column_stack((u.astype(np.int32), v.astype(np.int32)))
    return PointSet(points, pixels, valid_ratio)


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int,
    threshold_m: float,
    rng: np.random.Generator,
) -> Optional[PlaneModel]:
    """Fit a plane with three-point RANSAC, then refine it with SVD."""

    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return None

    best_mask: Optional[np.ndarray] = None
    best_count = 0
    best_rms = float("inf")

    for _ in range(max(1, iterations)):
        sample_indices = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[sample_indices]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        d = -float(np.dot(normal, p0))
        residuals = np.abs(points @ normal + d)
        mask = residuals <= threshold_m
        count = int(mask.sum())
        if count < 3:
            continue
        rms = float(np.sqrt(np.mean(np.square(residuals[mask]))))
        if count > best_count or (count == best_count and rms < best_rms):
            best_count = count
            best_rms = rms
            best_mask = mask
            if count >= int(0.98 * len(points)):
                break

    if best_mask is None or best_count < 3:
        return None

    inliers = points[best_mask]
    centroid = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    normal /= np.linalg.norm(normal)
    if normal[1] > 0:  # Deterministic upward normal in RealSense coordinates.
        normal = -normal
    d = -float(np.dot(normal, centroid))

    # One final inlier/refinement pass reduces bias from the first random model.
    residuals = np.abs(points @ normal + d)
    refined_mask = residuals <= threshold_m
    if int(refined_mask.sum()) >= 3:
        inliers = points[refined_mask]
        centroid = np.mean(inliers, axis=0)
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal = vh[-1].astype(np.float64)
        normal /= np.linalg.norm(normal)
        if normal[1] > 0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        residuals = np.abs(points @ normal + d)
        refined_mask = residuals <= threshold_m

    inlier_count = int(refined_mask.sum())
    if inlier_count < 3:
        return None
    rms_m = float(np.sqrt(np.mean(np.square(residuals[refined_mask]))))
    return PlaneModel(
        normal=normal,
        d=d,
        inlier_ratio=inlier_count / float(len(points)),
        rms_m=rms_m,
        inlier_count=inlier_count,
    )


def robust_height_cluster(
    distances_m: np.ndarray,
    config: DetectorConfig,
) -> Optional[Tuple[float, float, float, np.ndarray]]:
    """Find the densest plausible height cluster and return median/MAD/support."""

    plausible = distances_m[
        np.isfinite(distances_m)
        & (distances_m >= config.min_step_height_m)
        & (distances_m <= config.max_step_height_m)
    ]
    if plausible.size < config.min_top_points:
        return None

    bin_width = max(config.height_histogram_bin_m, 0.001)
    bin_count = max(
        1,
        int(math.ceil(
            (config.max_step_height_m - config.min_step_height_m) / bin_width
        )),
    )
    counts, edges = np.histogram(
        plausible,
        bins=bin_count,
        range=(config.min_step_height_m, config.max_step_height_m),
    )
    best_index = int(np.argmax(counts))
    center = 0.5 * (edges[best_index] + edges[best_index + 1])
    cluster_mask = np.abs(plausible - center) <= config.height_cluster_radius_m
    cluster = plausible[cluster_mask]
    if cluster.size < config.min_top_points:
        return None

    height_m = float(np.median(cluster))
    mad_m = float(np.median(np.abs(cluster - height_m)))
    cluster_ratio = float(cluster.size / plausible.size)
    return height_m, mad_m, cluster_ratio, cluster


def line_angle_deg(line: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    return abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))


def line_length(line: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)


def _finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


class RealSenseSource:
    """D435 depth+colour source aligned into the native depth coordinate grid."""

    def __init__(self, config: CameraConfig):
        require_runtime_dependencies()
        self.config = config
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.depth)
        self.profile = None
        self.depth_scale = 0.001
        self.spatial = None
        self.temporal = None

    def start(self) -> None:
        stream_config = rs.config()
        stream_config.enable_stream(
            rs.stream.depth,
            self.config.depth_width,
            self.config.depth_height,
            rs.format.z16,
            self.config.fps,
        )
        stream_config.enable_stream(
            rs.stream.color,
            self.config.color_width,
            self.config.color_height,
            rs.format.bgr8,
            self.config.fps,
        )
        self.profile = self.pipeline.start(stream_config)
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        if (
            self.config.emitter_enabled
            and depth_sensor.supports(rs.option.emitter_enabled)
        ):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0)

        if self.config.use_filters:
            self.spatial = rs.spatial_filter()
            self.spatial.set_option(rs.option.filter_magnitude, 2.0)
            self.spatial.set_option(rs.option.filter_smooth_alpha, 0.50)
            self.spatial.set_option(rs.option.filter_smooth_delta, 20.0)
            self.temporal = rs.temporal_filter()
            self.temporal.set_option(rs.option.filter_smooth_alpha, 0.40)
            self.temporal.set_option(rs.option.filter_smooth_delta, 20.0)

        for _ in range(max(0, self.config.warmup_frames)):
            self.pipeline.wait_for_frames(timeout_ms=5000)

    def read(self) -> Tuple[np.ndarray, np.ndarray, Any, float]:
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("정렬된 Depth/Color 프레임을 받지 못했습니다.")

        filtered_depth = depth_frame
        if self.spatial is not None:
            filtered_depth = self.spatial.process(filtered_depth)
        if self.temporal is not None:
            filtered_depth = self.temporal.process(filtered_depth)

        depth_raw = np.asanyarray(filtered_depth.get_data())
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        color_image = np.asanyarray(color_frame.get_data()).copy()
        if color_image.shape[:2] != depth_m.shape[:2]:
            raise RuntimeError(
                "Color→Depth 정렬 후 두 영상 크기가 다릅니다: "
                f"color={color_image.shape[:2]}, depth={depth_m.shape[:2]}"
            )

        intrinsics = (
            filtered_depth.profile.as_video_stream_profile().get_intrinsics()
        )
        timestamp_s = float(depth_frame.get_timestamp()) / 1000.0
        return color_image, depth_m, intrinsics, timestamp_s

    def stop(self) -> None:
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None

    def __enter__(self) -> "RealSenseSource":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


class AutoStepDetector:
    """Canny/Hough proposals validated by 3D floor-relative height."""

    def __init__(
        self,
        camera_config: CameraConfig,
        detector_config: DetectorConfig,
    ):
        if cv2 is None:
            raise RuntimeError("자동 검출에는 opencv-python이 필요합니다.")
        self.camera_config = camera_config
        self.config = detector_config
        self.rng = np.random.default_rng(detector_config.random_seed)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _edge_map(self, color_image: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        median_intensity = float(np.median(gray))
        low = int(max(30, 0.66 * median_intensity))
        high = int(min(220, max(low + 30, 1.33 * median_intensity)))
        color_edges = cv2.Canny(gray, low, high)

        valid_pairs = (
            np.isfinite(depth_m[:-1])
            & np.isfinite(depth_m[1:])
            & (depth_m[:-1] >= self.camera_config.min_depth_m)
            & (depth_m[1:] >= self.camera_config.min_depth_m)
            & (depth_m[:-1] <= self.camera_config.max_depth_m)
            & (depth_m[1:] <= self.camera_config.max_depth_m)
        )
        vertical_jump = np.zeros_like(depth_m, dtype=np.uint8)
        jump = np.abs(depth_m[1:] - depth_m[:-1])
        vertical_jump[1:][
            valid_pairs & (jump >= self.config.depth_edge_threshold_m)
        ] = 255
        vertical_jump = cv2.dilate(
            vertical_jump,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        combined = cv2.bitwise_or(color_edges, vertical_jump)
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
        )

        h, w = combined.shape
        search = ratio_roi(
            w,
            h,
            self.config.search_x_min_ratio,
            self.config.search_y_min_ratio,
            self.config.search_x_max_ratio,
            self.config.search_y_max_ratio,
        )
        mask = np.zeros_like(combined)
        x, y, rw, rh = search
        mask[y : y + rh, x : x + rw] = 255
        return cv2.bitwise_and(combined, mask)

    def _line_candidates(
        self,
        edge_map: np.ndarray,
    ) -> Sequence[Tuple[int, int, int, int]]:
        h, w = edge_map.shape
        minimum_length = max(20, int(w * self.config.min_line_length_ratio))
        lines = cv2.HoughLinesP(
            edge_map,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.config.hough_threshold,
            minLineLength=minimum_length,
            maxLineGap=self.config.hough_max_gap_px,
        )
        if lines is None:
            return []

        # OpenCV builds may return HoughLinesP results as either (N, 1, 4)
        # or (N, 4).  Normalize both forms before iterating.  Indexing with
        # ``lines[:, 0, :]`` crashes on the latter form.
        normalized_lines = np.asarray(lines).reshape(-1, 4)

        candidates = []
        for raw in normalized_lines:
            x1, y1, x2, y2 = (int(value) for value in raw)
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            line = (x1, y1, x2, y2)
            angle = line_angle_deg(line)
            length = line_length(line)
            midpoint_x = 0.5 * (x1 + x2)
            if angle > self.config.max_line_angle_deg:
                continue
            if length < minimum_length:
                continue
            if not (0.12 * w <= midpoint_x <= 0.88 * w):
                continue
            candidates.append(line)

        candidates.sort(key=line_length, reverse=True)
        deduplicated = []
        for candidate in candidates:
            cy = 0.5 * (candidate[1] + candidate[3])
            cx = 0.5 * (candidate[0] + candidate[2])
            is_duplicate = False
            for kept in deduplicated:
                ky = 0.5 * (kept[1] + kept[3])
                kx = 0.5 * (kept[0] + kept[2])
                if abs(cy - ky) <= 7 and abs(cx - kx) <= 0.20 * w:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(candidate)
            if len(deduplicated) >= self.config.max_line_candidates:
                break
        return deduplicated

    def _candidate_rois(
        self,
        line: Tuple[int, int, int, int],
        width: int,
        height: int,
        floor_roi: ROI,
    ) -> Tuple[ROI, ROI]:
        x1, y1, x2, y2 = line
        edge_y = int(round(0.5 * (y1 + y2)))
        margin = self.config.horizontal_margin_px
        left = max(x1 + margin, int(0.08 * width))
        right = min(x2 - margin, int(0.92 * width))
        if right <= left:
            return (0, 0, 0, 0), (0, 0, 0, 0)

        top_bottom = edge_y - self.config.edge_exclusion_px
        top_y = top_bottom - self.config.top_band_height_px
        top_roi = clip_roi(
            (left, top_y, right - left, self.config.top_band_height_px),
            width,
            height,
        )

        face_y = edge_y + self.config.edge_exclusion_px
        face_roi = clip_roi(
            (left, face_y, right - left, self.config.face_band_height_px),
            width,
            height,
        )
        return top_roi, face_roi

    def _measure_candidate(
        self,
        line: Tuple[int, int, int, int],
        depth_m: np.ndarray,
        intrinsics: Any,
        floor_roi: ROI,
        floor_plane: PlaneModel,
    ) -> Optional[Measurement]:
        image_h, image_w = depth_m.shape
        top_roi, face_roi = self._candidate_rois(
            line, image_w, image_h, floor_roi
        )
        if top_roi[2] * top_roi[3] <= 0:
            return None

        top_set = points_from_roi(
            depth_m,
            intrinsics,
            top_roi,
            self.camera_config.min_depth_m,
            self.camera_config.max_depth_m,
            self.config.max_top_points,
        )
        if len(top_set.points) < self.config.min_top_points:
            return None

        height_result = robust_height_cluster(
            floor_plane.distances(top_set.points), self.config
        )
        if height_result is None:
            return None
        height_m, top_mad_m, cluster_ratio, _ = height_result
        if cluster_ratio < self.config.min_top_cluster_ratio:
            return None
        if top_mad_m > self.config.max_top_mad_m:
            return None

        face_set = points_from_roi(
            depth_m,
            intrinsics,
            face_roi,
            self.camera_config.min_depth_m,
            self.camera_config.max_depth_m,
            self.config.max_top_points,
        )
        face_support = np.empty((0, 3), dtype=np.float64)
        if len(face_set.points):
            face_heights = floor_plane.distances(face_set.points)
            upper_limit = max(0.02, height_m - 0.006)
            face_mask = (
                (face_heights >= 0.010)
                & (face_heights <= upper_limit)
            )
            face_support = face_set.points[face_mask]

        if len(face_support) >= self.config.min_face_points:
            distance_m = float(np.median(face_support[:, 2]))
            center_offset_m = float(np.median(face_support[:, 0]))
        else:
            # Fallback uses the nearest portion of the top patch, not the edge.
            top_z_limit = float(np.quantile(top_set.points[:, 2], 0.35))
            near_top = top_set.points[top_set.points[:, 2] <= top_z_limit]
            distance_m = float(np.median(near_top[:, 2]))
            center_offset_m = float(np.median(near_top[:, 0]))

        distance_m = max(
            0.0, distance_m - self.config.camera_forward_offset_m
        )

        length_score = min(1.0, line_length(line) / (0.55 * image_w))
        angle_score = max(
            0.0,
            1.0 - line_angle_deg(line) / self.config.max_line_angle_deg,
        )
        center_x = 0.5 * (line[0] + line[2])
        center_score = max(0.0, 1.0 - abs(center_x - 0.5 * image_w) / (0.5 * image_w))
        floor_score = np.clip(
            (floor_plane.inlier_ratio - self.config.min_floor_inlier_ratio)
            / max(1e-6, 1.0 - self.config.min_floor_inlier_ratio),
            0.0,
            1.0,
        )
        cluster_score = np.clip(
            (cluster_ratio - self.config.min_top_cluster_ratio)
            / max(1e-6, 1.0 - self.config.min_top_cluster_ratio),
            0.0,
            1.0,
        )
        spread_score = np.clip(
            1.0 - top_mad_m / max(1e-6, self.config.max_top_mad_m),
            0.0,
            1.0,
        )
        valid_score = np.clip(top_set.valid_ratio / 0.80, 0.0, 1.0)
        confidence = float(
            0.16 * length_score
            + 0.12 * angle_score
            + 0.10 * center_score
            + 0.20 * floor_score
            + 0.20 * cluster_score
            + 0.14 * spread_score
            + 0.08 * valid_score
        )

        return Measurement(
            height_m=height_m,
            distance_m=distance_m,
            center_offset_m=center_offset_m,
            confidence=confidence,
            edge_y_px=0.5 * (line[1] + line[3]),
            line=line,
            top_roi=top_roi,
            floor_roi=floor_roi,
            face_roi=face_roi,
            floor_inlier_ratio=floor_plane.inlier_ratio,
            floor_rms_m=floor_plane.rms_m,
            top_mad_m=top_mad_m,
            top_cluster_ratio=cluster_ratio,
            source="AUTO",
        )

    def detect(
        self,
        color_image: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: Any,
    ) -> Tuple[Optional[Measurement], DetectionDebug]:
        image_h, image_w = depth_m.shape
        floor_roi = ratio_roi(
            image_w,
            image_h,
            self.config.floor_x_min_ratio,
            self.config.floor_y_min_ratio,
            self.config.floor_x_max_ratio,
            self.config.floor_y_max_ratio,
        )
        debug = DetectionDebug(floor_roi=floor_roi)

        floor_set = points_from_roi(
            depth_m,
            intrinsics,
            floor_roi,
            self.camera_config.min_depth_m,
            self.camera_config.max_depth_m,
            self.config.max_floor_points,
        )
        debug.floor_valid_points = len(floor_set.points)
        debug.floor_valid_ratio = floor_set.valid_ratio
        if len(floor_set.points) < self.config.min_floor_points:
            debug.reason = (
                "바닥 유효점 부족 "
                f"({len(floor_set.points)}/{self.config.min_floor_points}, "
                f"valid={floor_set.valid_ratio * 100:.1f}%)"
            )
            return None, debug

        floor_plane = fit_plane_ransac(
            floor_set.points,
            self.config.ransac_iterations,
            self.config.ransac_threshold_m,
            self.rng,
        )
        debug.floor_plane = floor_plane
        if floor_plane is None:
            debug.reason = "바닥 평면 추정 실패"
            return None, debug
        if floor_plane.inlier_ratio < self.config.min_floor_inlier_ratio:
            debug.reason = "바닥 평면 inlier 비율 부족"
            return None, debug

        edge_map = self._edge_map(color_image, depth_m)
        debug.edge_map = edge_map
        lines = self._line_candidates(edge_map)
        debug.candidate_count = len(lines)
        if not lines:
            debug.reason = "수평 경계 후보 없음"
            return None, debug

        measurements = []
        for line in lines:
            measurement = self._measure_candidate(
                line, depth_m, intrinsics, floor_roi, floor_plane
            )
            if measurement is not None:
                measurements.append(measurement)

        if not measurements:
            debug.reason = "3D 높이 조건을 통과한 경계 없음"
            return None, debug

        best = max(measurements, key=lambda item: item.confidence)
        if best.confidence < self.config.min_candidate_confidence:
            debug.reason = "최고 후보 신뢰도 부족"
            return best, debug
        debug.reason = "정상"
        return best, debug


class ManualHeightEstimator:
    """RANSAC floor plane + manual top/floor rectangles."""

    def __init__(
        self,
        camera_config: CameraConfig,
        detector_config: DetectorConfig,
    ):
        self.camera_config = camera_config
        self.config = detector_config
        self.rng = np.random.default_rng(detector_config.random_seed + 1)

    def measure(
        self,
        depth_m: np.ndarray,
        intrinsics: Any,
        top_roi: ROI,
        floor_roi: ROI,
    ) -> Optional[Measurement]:
        floor_set = points_from_roi(
            depth_m,
            intrinsics,
            floor_roi,
            self.camera_config.min_depth_m,
            self.camera_config.max_depth_m,
            self.config.max_floor_points,
        )
        top_set = points_from_roi(
            depth_m,
            intrinsics,
            top_roi,
            self.camera_config.min_depth_m,
            self.camera_config.max_depth_m,
            self.config.max_top_points,
        )
        if (
            len(floor_set.points) < self.config.min_floor_points
            or len(top_set.points) < self.config.min_top_points
        ):
            return None

        plane = fit_plane_ransac(
            floor_set.points,
            self.config.ransac_iterations,
            self.config.ransac_threshold_m,
            self.rng,
        )
        if plane is None or plane.inlier_ratio < self.config.min_floor_inlier_ratio:
            return None

        result = robust_height_cluster(plane.distances(top_set.points), self.config)
        if result is None:
            return None
        height_m, mad_m, cluster_ratio, _ = result
        if (
            cluster_ratio < self.config.min_top_cluster_ratio
            or mad_m > self.config.max_top_mad_m
        ):
            return None

        # A manual top ROI has no face rectangle. Use its nearest third as a
        # conservative approximation of the top-front distance instead of the
        # full-patch median, which is biased toward the rear of a deep step.
        near_limit = float(np.quantile(top_set.points[:, 2], 0.35))
        near_top = top_set.points[top_set.points[:, 2] <= near_limit]
        distance_m = float(np.median(near_top[:, 2]))
        distance_m = max(
            0.0, distance_m - self.config.camera_forward_offset_m
        )
        center_offset_m = float(np.median(top_set.points[:, 0]))
        floor_score = np.clip(
            (plane.inlier_ratio - self.config.min_floor_inlier_ratio)
            / max(1e-6, 1.0 - self.config.min_floor_inlier_ratio),
            0.0,
            1.0,
        )
        cluster_score = np.clip(
            (cluster_ratio - self.config.min_top_cluster_ratio)
            / max(1e-6, 1.0 - self.config.min_top_cluster_ratio),
            0.0,
            1.0,
        )
        spread_score = np.clip(
            1.0 - mad_m / max(1e-6, self.config.max_top_mad_m),
            0.0,
            1.0,
        )
        confidence = float(
            0.35 * floor_score + 0.35 * cluster_score + 0.30 * spread_score
        )
        edge_y = float(top_roi[1] + top_roi[3])
        return Measurement(
            height_m=height_m,
            distance_m=distance_m,
            center_offset_m=center_offset_m,
            confidence=confidence,
            edge_y_px=edge_y,
            line=None,
            top_roi=top_roi,
            floor_roi=floor_roi,
            face_roi=None,
            floor_inlier_ratio=plane.inlier_ratio,
            floor_rms_m=plane.rms_m,
            top_mad_m=mad_m,
            top_cluster_ratio=cluster_ratio,
            source="MANUAL",
        )


class MeasurementStabilizer:
    """Reject track switches and median-filter 15–30 valid frames."""

    def __init__(self, config: StabilityConfig):
        self.config = config
        self.history: Deque[Measurement] = deque(maxlen=config.window_frames)
        self.misses = 0

    def reset(self) -> None:
        self.history.clear()
        self.misses = 0

    def update(self, measurement: Optional[Measurement]) -> StableSummary:
        if (
            measurement is None
            or measurement.confidence < self.config.min_confidence
        ):
            self.misses += 1
            if self.misses >= self.config.reset_after_misses:
                self.reset()
            return self.summary()

        self.misses = 0
        if self.history:
            height_median = float(
                np.median([item.height_m for item in self.history])
            )
            edge_median = float(
                np.median([item.edge_y_px for item in self.history])
            )
            if (
                abs(measurement.height_m - height_median)
                > self.config.max_tracking_height_jump_m
                or abs(measurement.edge_y_px - edge_median)
                > self.config.max_tracking_edge_jump_px
            ):
                self.reset()

        self.history.append(measurement)
        return self.summary()

    def summary(self) -> StableSummary:
        if not self.history:
            return StableSummary(
                False,
                0,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )

        heights = np.asarray([item.height_m for item in self.history])
        edges = np.asarray([item.edge_y_px for item in self.history])
        confidences = np.asarray([item.confidence for item in self.history])
        height_spread = float(np.percentile(heights, 90) - np.percentile(heights, 10))
        edge_spread = float(np.percentile(edges, 90) - np.percentile(edges, 10))
        stable = (
            len(self.history) >= self.config.required_frames
            and height_spread <= self.config.max_height_spread_m
            and edge_spread <= self.config.max_edge_spread_px
            and float(np.median(confidences)) >= self.config.min_confidence
        )
        return StableSummary(
            stable=stable,
            samples=len(self.history),
            height_m=float(np.median(heights)),
            distance_m=_finite_median(item.distance_m for item in self.history),
            center_offset_m=_finite_median(
                item.center_offset_m for item in self.history
            ),
            height_spread_m=height_spread,
            edge_spread_px=edge_spread,
            confidence=float(np.median(confidences)),
        )


def parse_roi(text: str) -> ROI:
    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError("ROI 형식은 x,y,width,height 입니다.") from exc
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise ValueError("ROI 형식은 x,y,width,height이며 폭과 높이는 양수여야 합니다.")
    return values  # type: ignore[return-value]


def make_depth_colormap(
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("화면 표시에 opencv-python이 필요합니다.")
    clipped = np.clip(depth_m, min_depth_m, max_depth_m)
    normalized = (
        255.0 * (clipped - min_depth_m) / max(1e-6, max_depth_m - min_depth_m)
    )
    normalized[~np.isfinite(depth_m) | (depth_m <= 0)] = 0
    gray = normalized.astype(np.uint8)
    colour = cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)
    colour[~np.isfinite(depth_m) | (depth_m <= 0)] = (0, 0, 0)
    return colour


def draw_roi(
    image: np.ndarray,
    roi: ROI,
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    if cv2 is None:
        return
    x, y, w, h = clip_roi(roi, image.shape[1], image.shape[0])
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(
        image,
        label,
        (x + 4, max(18, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_measurement(
    image: np.ndarray,
    measurement: Optional[Measurement],
    color: Tuple[int, int, int],
) -> None:
    if cv2 is None or measurement is None:
        return
    if measurement.line is not None:
        x1, y1, x2, y2 = measurement.line
        cv2.line(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    draw_roi(image, measurement.top_roi, color, f"{measurement.source} TOP")
    floor_color = (255, 160, 0) if measurement.source == "AUTO" else (180, 80, 255)
    draw_roi(image, measurement.floor_roi, floor_color, f"{measurement.source} FLOOR")
    if measurement.face_roi is not None:
        draw_roi(image, measurement.face_roi, (0, 200, 255), "FACE", 1)


def put_text_lines(
    image: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int] = (14, 28),
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    if cv2 is None:
        return
    x, y = origin
    line_height = 25
    panel_width = min(
        image.shape[1] - x - 6,
        max(300, max((len(line) for line in lines), default=1) * 10),
    )
    panel_height = line_height * len(lines) + 12
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x - 8, y - 22),
        (x + panel_width, y - 22 + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )


def compose_side_by_side(
    color_view: np.ndarray,
    depth_m: np.ndarray,
    camera_config: CameraConfig,
) -> np.ndarray:
    depth_view = make_depth_colormap(
        depth_m, camera_config.min_depth_m, camera_config.max_depth_m
    )
    cv2.putText(
        depth_view,
        "DEPTH",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.hstack((color_view, depth_view))


def append_measurements_csv(
    path: Path,
    entries: Sequence[Tuple[str, StableSummary, Optional[Measurement]]],
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    fieldnames = [
        "saved_at",
        "mode",
        "stable",
        "samples",
        "height_cm",
        "distance_cm",
        "center_offset_cm",
        "height_p90_p10_cm",
        "confidence",
        "floor_inlier_ratio",
        "floor_rms_mm",
        "top_mad_mm",
        "top_cluster_ratio",
    ]
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for mode, summary, latest in entries:
            writer.writerow(
                {
                    "saved_at": timestamp,
                    "mode": mode,
                    "stable": summary.stable,
                    "samples": summary.samples,
                    "height_cm": _format_number(summary.height_m * 100.0, 3),
                    "distance_cm": _format_number(summary.distance_m * 100.0, 3),
                    "center_offset_cm": _format_number(
                        summary.center_offset_m * 100.0, 3
                    ),
                    "height_p90_p10_cm": _format_number(
                        summary.height_spread_m * 100.0, 3
                    ),
                    "confidence": _format_number(summary.confidence, 3),
                    "floor_inlier_ratio": _format_number(
                        latest.floor_inlier_ratio if latest else float("nan"), 3
                    ),
                    "floor_rms_mm": _format_number(
                        latest.floor_rms_m * 1000.0 if latest else float("nan"), 3
                    ),
                    "top_mad_mm": _format_number(
                        latest.top_mad_m * 1000.0 if latest else float("nan"), 3
                    ),
                    "top_cluster_ratio": _format_number(
                        latest.top_cluster_ratio if latest else float("nan"), 3
                    ),
                }
            )


def _format_number(value: float, digits: int) -> str:
    return "" if not np.isfinite(value) else f"{value:.{digits}f}"


class RateMeter:
    def __init__(self, window: int = 30):
        self.times: Deque[float] = deque(maxlen=max(2, window))

    def tick(self) -> float:
        now = time.perf_counter()
        self.times.append(now)
        if len(self.times) < 2:
            return float("nan")
        duration = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / duration if duration > 0 else float("nan")


def summary_lines(
    title: str,
    measurement: Optional[Measurement],
    summary: StableSummary,
    fps: float,
    reason: str = "",
) -> Sequence[str]:
    if measurement is None:
        current = "current: --"
    else:
        current = (
            f"current: H={measurement.height_m * 100:.2f} cm  "
            f"D={measurement.distance_m * 100:.1f} cm  "
            f"conf={measurement.confidence:.2f}"
        )
    if summary.samples:
        stable_label = "STABLE" if summary.stable else "collecting"
        filtered = (
            f"{stable_label}: H={summary.height_m * 100:.2f} cm  "
            f"spread={summary.height_spread_m * 100:.2f} cm  "
            f"n={summary.samples}"
        )
    else:
        filtered = "collecting: n=0"
    fps_text = "--" if not np.isfinite(fps) else f"{fps:.1f}"
    output = [title, current, filtered, f"processing FPS: {fps_text}"]
    if reason:
        output.append(f"status: {reason}")
    return output


def stable_console_text(label: str, summary: StableSummary) -> str:
    return (
        f"[{label}] 안정화 완료 | 높이={summary.height_m * 100:.2f} cm | "
        f"거리={summary.distance_m * 100:.1f} cm | "
        f"P90-P10={summary.height_spread_m * 100:.2f} cm | "
        f"신뢰도={summary.confidence:.2f} | 프레임={summary.samples}"
    )
