#!/usr/bin/env python3
"""Automatic D435 step detection and floor-relative height measurement.

Pipeline:
    aligned RGB/depth -> Canny + depth discontinuity -> Hough line proposals
    -> RANSAC foreground-floor plane -> top-point perpendicular distances
    -> robust per-frame median -> temporal median over 15–30 valid frames

No mouse-selected ROI is used.  The broad search/floor bands are fixed ratios
of the image and may be tuned with command-line options.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from step_height_core import (
    AutoStepDetector,
    CameraConfig,
    DetectorConfig,
    MeasurementStabilizer,
    RateMeter,
    RealSenseSource,
    StabilityConfig,
    append_measurements_csv,
    compose_side_by_side,
    cv2,
    draw_measurement,
    draw_roi,
    put_text_lines,
    stable_console_text,
    summary_lines,
)


WINDOW_NAME = "D435 Automatic Step Height"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D435 자동 단차 검출 및 바닥 평면 기준 높이 측정"
    )
    parser.add_argument("--depth-width", type=int, default=848)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=2.00)
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="RealSense spatial/temporal filter를 끕니다.",
    )
    parser.add_argument(
        "--no-emitter",
        action="store_true",
        help="D435 적외선 emitter를 코드에서 켜지 않습니다.",
    )

    parser.add_argument("--min-height-cm", type=float, default=1.5)
    parser.add_argument("--max-height-cm", type=float, default=25.0)
    parser.add_argument("--ransac-threshold-mm", type=float, default=6.0)
    parser.add_argument("--min-floor-inlier", type=float, default=0.55)
    parser.add_argument("--max-line-angle-deg", type=float, default=8.0)
    parser.add_argument("--min-line-length-ratio", type=float, default=0.22)
    parser.add_argument("--search-y-min", type=float, default=0.12)
    parser.add_argument("--search-y-max", type=float, default=0.88)
    parser.add_argument(
        "--floor-y-min",
        type=float,
        default=0.66,
        help="자동 바닥 평면 탐색 시작 높이 비율(기본 0.66)",
    )
    parser.add_argument(
        "--floor-y-max",
        type=float,
        default=0.90,
        help="자동 바닥 평면 탐색 종료 높이 비율(기본 0.90)",
    )
    parser.add_argument(
        "--camera-forward-offset-cm",
        type=float,
        default=0.0,
        help="카메라 렌즈에서 로봇 전면까지의 앞뒤 오프셋(거리 출력에서 차감)",
    )

    parser.add_argument("--stability-window", type=int, default=30)
    parser.add_argument("--required-frames", type=int, default=15)
    parser.add_argument("--max-height-spread-cm", type=float, default=1.0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("step_measurements.csv"),
        help="s 키 또는 --auto-save로 저장할 CSV 경로",
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="처음 STABLE이 되는 순간 CSV에 자동 저장",
    )
    parser.add_argument(
        "--exit-when-stable",
        action="store_true",
        help="안정화된 결과를 출력/저장한 뒤 종료",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="OpenCV 창 없이 터미널에서 실행",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    ratios = [
        args.search_y_min,
        args.search_y_max,
        args.floor_y_min,
        args.floor_y_max,
    ]
    if any(value < 0.0 or value > 1.0 for value in ratios):
        parser.error("검색/바닥 비율은 0.0~1.0이어야 합니다.")
    if args.search_y_min >= args.search_y_max:
        parser.error("--search-y-min은 --search-y-max보다 작아야 합니다.")
    if args.floor_y_min >= args.floor_y_max:
        parser.error("--floor-y-min은 --floor-y-max보다 작아야 합니다.")
    if args.min_height_cm <= 0 or args.min_height_cm >= args.max_height_cm:
        parser.error("최소 단차 높이는 0보다 크고 최대 높이보다 작아야 합니다.")
    if not (3 <= args.required_frames <= args.stability_window):
        parser.error("required-frames는 3 이상, stability-window 이하여야 합니다.")


def make_configs(
    args: argparse.Namespace,
) -> tuple[CameraConfig, DetectorConfig, StabilityConfig]:
    camera = CameraConfig(
        depth_width=args.depth_width,
        depth_height=args.depth_height,
        color_width=args.color_width,
        color_height=args.color_height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        use_filters=not args.no_filters,
        emitter_enabled=not args.no_emitter,
    )
    detector = DetectorConfig(
        search_y_min_ratio=args.search_y_min,
        search_y_max_ratio=args.search_y_max,
        floor_y_min_ratio=args.floor_y_min,
        floor_y_max_ratio=args.floor_y_max,
        max_line_angle_deg=args.max_line_angle_deg,
        min_line_length_ratio=args.min_line_length_ratio,
        min_step_height_m=args.min_height_cm / 100.0,
        max_step_height_m=args.max_height_cm / 100.0,
        ransac_threshold_m=args.ransac_threshold_mm / 1000.0,
        min_floor_inlier_ratio=args.min_floor_inlier,
        camera_forward_offset_m=args.camera_forward_offset_cm / 100.0,
    )
    stability = StabilityConfig(
        window_frames=args.stability_window,
        required_frames=args.required_frames,
        max_height_spread_m=args.max_height_spread_cm / 100.0,
        min_confidence=detector.min_candidate_confidence,
    )
    return camera, detector, stability


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    camera_config, detector_config, stability_config = make_configs(args)

    detector = AutoStepDetector(camera_config, detector_config)
    stabilizer = MeasurementStabilizer(stability_config)
    rate_meter = RateMeter()
    show_depth = True
    stable_announced = False
    last_console_time = 0.0
    last_measurement = None

    print("D435 자동 측정을 시작합니다. 카메라와 단차를 정지시켜 주세요.")
    if not args.headless:
        print("키: q/ESC 종료 | r 누적 초기화 | s 안정값 CSV 저장 | d Depth 화면 전환")

    try:
        with RealSenseSource(camera_config) as source:
            while True:
                color_image, depth_m, intrinsics, _ = source.read()
                measurement, debug = detector.detect(
                    color_image, depth_m, intrinsics
                )
                summary = stabilizer.update(measurement)
                fps = rate_meter.tick()
                last_measurement = measurement

                if summary.stable and not stable_announced:
                    print(stable_console_text("AUTO", summary))
                    if args.auto_save:
                        append_measurements_csv(
                            args.csv,
                            [("AUTO", summary, last_measurement)],
                        )
                        print(f"CSV 저장: {args.csv.expanduser().resolve()}")
                    stable_announced = True
                    if args.exit_when_stable:
                        break
                elif summary.samples == 0:
                    stable_announced = False

                now = time.monotonic()
                if args.headless and now - last_console_time >= 1.0:
                    if measurement is None:
                        print(f"[AUTO] 검출 대기 | {debug.reason}")
                    else:
                        print(
                            f"[AUTO] 현재 {measurement.height_m * 100:.2f} cm | "
                            f"누적 {summary.samples}/{stability_config.required_frames} | "
                            f"신뢰도 {measurement.confidence:.2f}"
                        )
                    last_console_time = now

                if args.headless:
                    continue

                view = color_image.copy()
                if debug.floor_roi is not None and measurement is None:
                    draw_roi(view, debug.floor_roi, (255, 160, 0), "FLOOR SEARCH", 1)
                draw_measurement(view, measurement, (0, 255, 0))
                lines = list(
                    summary_lines(
                        "AUTO / floor-plane perpendicular height",
                        measurement,
                        summary,
                        fps,
                        debug.reason,
                    )
                )
                lines.append("keys: q quit | r reset | s save STABLE | d depth view")
                put_text_lines(view, lines)

                display = (
                    compose_side_by_side(view, depth_m, camera_config)
                    if show_depth
                    else view
                )
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    stabilizer.reset()
                    stable_announced = False
                    print("AUTO 누적값을 초기화했습니다.")
                elif key == ord("d"):
                    show_depth = not show_depth
                elif key == ord("s"):
                    if not summary.stable:
                        print("아직 STABLE이 아닙니다. 카메라를 고정하고 더 기다려 주세요.")
                    else:
                        append_measurements_csv(
                            args.csv,
                            [("AUTO", summary, last_measurement)],
                        )
                        print(f"CSV 저장: {args.csv.expanduser().resolve()}")
    except KeyboardInterrupt:
        print("\n사용자가 종료했습니다.")
    except RuntimeError as exc:
        print(f"실행 오류: {exc}", file=sys.stderr)
        return 1
    finally:
        if cv2 is not None and not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
