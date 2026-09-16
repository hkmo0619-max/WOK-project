#!/usr/bin/env python3
"""Manual, automatic, and same-frame comparison modes for D435 step height.

Manual mode asks for a top-surface rectangle and a foreground-floor rectangle.
Automatic mode uses the Canny/Hough + 3D validation pipeline. Compare mode runs
both estimators on the same aligned depth frames and shows their stable-height
difference. Both estimators use the same RANSAC plane-distance calculation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from automatic_step_height import (
    build_parser as build_auto_parser,
    make_configs,
    validate_args as validate_auto_args,
)
from step_height_core import (
    AutoStepDetector,
    ManualHeightEstimator,
    Measurement,
    MeasurementStabilizer,
    RateMeter,
    RealSenseSource,
    ROI,
    StableSummary,
    append_measurements_csv,
    compose_side_by_side,
    cv2,
    draw_measurement,
    draw_roi,
    parse_roi,
    put_text_lines,
    stable_console_text,
)


WINDOW_NAME = "D435 Manual + Automatic Step Height"


def build_parser() -> argparse.ArgumentParser:
    parser = build_auto_parser()
    parser.description = "D435 수동·자동 단차 높이 측정 및 같은 프레임 비교"
    parser.add_argument(
        "--mode",
        choices=("manual", "auto", "compare"),
        default="compare",
        help="manual=수동 ROI, auto=자동, compare=두 결과 동시 비교",
    )
    parser.add_argument(
        "--top-roi",
        type=str,
        default=None,
        metavar="X,Y,W,H",
        help="창 없이 또는 재사용할 수동 상면 ROI",
    )
    parser.add_argument(
        "--floor-roi",
        type=str,
        default=None,
        metavar="X,Y,W,H",
        help="창 없이 또는 재사용할 수동 바닥 ROI",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_auto_args(args, parser)
    if bool(args.top_roi) != bool(args.floor_roi):
        parser.error("--top-roi와 --floor-roi는 함께 지정해야 합니다.")
    if args.headless and args.mode in ("manual", "compare"):
        if not args.top_roi or not args.floor_roi:
            parser.error(
                "headless의 manual/compare 모드에는 --top-roi와 --floor-roi가 필요합니다."
            )


def select_manual_rois(color_image: np.ndarray) -> tuple[Optional[ROI], Optional[ROI]]:
    """Freeze one frame and collect top then floor rectangles."""

    print("1/2: 단차 상면 내부를 드래그하고 Enter 또는 Space를 누르세요.")
    print("     경계선에서 최소 10~20픽셀 떨어진 평탄한 부분을 선택하세요.")
    top = cv2.selectROI(
        "1 - Select STEP TOP (Enter=OK, c=Cancel)",
        color_image,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("1 - Select STEP TOP (Enter=OK, c=Cancel)")
    top_roi = tuple(int(value) for value in top)
    if top_roi[2] <= 0 or top_roi[3] <= 0:
        print("상면 ROI 선택을 취소했습니다.")
        return None, None

    print("2/2: 단차 앞쪽의 평탄한 바닥을 드래그하고 Enter 또는 Space를 누르세요.")
    print("     단차 수직면과 경계선이 들어가지 않게 선택하세요.")
    floor = cv2.selectROI(
        "2 - Select FOREGROUND FLOOR (Enter=OK, c=Cancel)",
        color_image,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("2 - Select FOREGROUND FLOOR (Enter=OK, c=Cancel)")
    floor_roi = tuple(int(value) for value in floor)
    if floor_roi[2] <= 0 or floor_roi[3] <= 0:
        print("바닥 ROI 선택을 취소했습니다.")
        return None, None
    return top_roi, floor_roi  # type: ignore[return-value]


def active_entries(
    mode: str,
    auto_summary: StableSummary,
    manual_summary: StableSummary,
    auto_measurement: Optional[Measurement],
    manual_measurement: Optional[Measurement],
) -> list[tuple[str, StableSummary, Optional[Measurement]]]:
    entries = []
    if mode in ("auto", "compare") and auto_summary.stable:
        entries.append(("AUTO", auto_summary, auto_measurement))
    if mode in ("manual", "compare") and manual_summary.stable:
        entries.append(("MANUAL", manual_summary, manual_measurement))
    return entries


def mode_is_ready(
    mode: str,
    auto_summary: StableSummary,
    manual_summary: StableSummary,
) -> bool:
    if mode == "auto":
        return auto_summary.stable
    if mode == "manual":
        return manual_summary.stable
    return auto_summary.stable and manual_summary.stable


def format_one_status(
    label: str,
    measurement: Optional[Measurement],
    summary: StableSummary,
) -> str:
    if measurement is None:
        return f"{label}: --"
    stable_label = "STABLE" if summary.stable else f"n={summary.samples}"
    return (
        f"{label}: now={measurement.height_m * 100:.2f} cm  "
        f"median={summary.height_m * 100:.2f} cm  "
        f"{stable_label}  conf={measurement.confidence:.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    camera_config, detector_config, stability_config = make_configs(args)

    try:
        top_roi = parse_roi(args.top_roi) if args.top_roi else None
        floor_roi = parse_roi(args.floor_roi) if args.floor_roi else None
    except ValueError as exc:
        parser.error(str(exc))

    auto_detector = AutoStepDetector(camera_config, detector_config)
    manual_estimator = ManualHeightEstimator(camera_config, detector_config)
    auto_stabilizer = MeasurementStabilizer(stability_config)
    manual_stabilizer = MeasurementStabilizer(stability_config)
    auto_summary = auto_stabilizer.summary()
    manual_summary = manual_stabilizer.summary()
    rate_meter = RateMeter()

    mode = args.mode
    show_depth = True
    ready_announced = False
    last_console_time = 0.0
    last_auto: Optional[Measurement] = None
    last_manual: Optional[Measurement] = None

    print(f"D435 측정을 시작합니다. 시작 모드: {mode.upper()}")
    if mode in ("manual", "compare") and top_roi is None:
        print("영상이 뜨면 m 키를 눌러 상면 ROI와 바닥 ROI를 한 번씩 선택하세요.")
    if not args.headless:
        print(
            "키: 1 MANUAL | 2 AUTO | 3 COMPARE | m ROI 선택 | c ROI 삭제 | "
            "r 누적 초기화 | s 저장 | d Depth 전환 | q 종료"
        )

    try:
        with RealSenseSource(camera_config) as source:
            while True:
                color_image, depth_m, intrinsics, _ = source.read()
                debug_reason = ""

                if mode in ("auto", "compare"):
                    last_auto, auto_debug = auto_detector.detect(
                        color_image, depth_m, intrinsics
                    )
                    auto_summary = auto_stabilizer.update(last_auto)
                    debug_reason = auto_debug.reason
                else:
                    auto_debug = None

                if mode in ("manual", "compare"):
                    if top_roi is not None and floor_roi is not None:
                        last_manual = manual_estimator.measure(
                            depth_m, intrinsics, top_roi, floor_roi
                        )
                    else:
                        last_manual = None
                    manual_summary = manual_stabilizer.update(last_manual)

                fps = rate_meter.tick()
                ready = mode_is_ready(mode, auto_summary, manual_summary)
                if ready and not ready_announced:
                    if mode in ("auto", "compare"):
                        print(stable_console_text("AUTO", auto_summary))
                    if mode in ("manual", "compare"):
                        print(stable_console_text("MANUAL", manual_summary))
                    if mode == "compare":
                        difference_cm = (
                            auto_summary.height_m - manual_summary.height_m
                        ) * 100.0
                        print(f"[COMPARE] AUTO - MANUAL = {difference_cm:+.2f} cm")
                    if args.auto_save:
                        entries = active_entries(
                            mode,
                            auto_summary,
                            manual_summary,
                            last_auto,
                            last_manual,
                        )
                        append_measurements_csv(args.csv, entries)
                        print(f"CSV 저장: {args.csv.expanduser().resolve()}")
                    ready_announced = True
                    if args.exit_when_stable:
                        break
                elif not ready:
                    ready_announced = False

                now = time.monotonic()
                if args.headless and now - last_console_time >= 1.0:
                    if mode in ("auto", "compare"):
                        if last_auto is None:
                            print(f"[AUTO] 검출 대기 | {debug_reason}")
                        else:
                            print(
                                f"[AUTO] {last_auto.height_m * 100:.2f} cm | "
                                f"n={auto_summary.samples} | conf={last_auto.confidence:.2f}"
                            )
                    if mode in ("manual", "compare"):
                        if last_manual is None:
                            print("[MANUAL] 유효 측정 대기")
                        else:
                            print(
                                f"[MANUAL] {last_manual.height_m * 100:.2f} cm | "
                                f"n={manual_summary.samples} | conf={last_manual.confidence:.2f}"
                            )
                    last_console_time = now

                if args.headless:
                    continue

                view = color_image.copy()
                if mode in ("auto", "compare"):
                    if auto_debug is not None and auto_debug.floor_roi is not None and last_auto is None:
                        draw_roi(
                            view,
                            auto_debug.floor_roi,
                            (255, 160, 0),
                            "AUTO FLOOR SEARCH",
                            1,
                        )
                    draw_measurement(view, last_auto, (0, 255, 0))
                if mode in ("manual", "compare"):
                    if top_roi is not None:
                        draw_roi(view, top_roi, (255, 0, 255), "MANUAL TOP")
                    if floor_roi is not None:
                        draw_roi(view, floor_roi, (180, 80, 255), "MANUAL FLOOR")

                fps_text = "--" if not np.isfinite(fps) else f"{fps:.1f}"
                lines = [f"MODE: {mode.upper()}  | processing FPS: {fps_text}"]
                if mode in ("auto", "compare"):
                    lines.append(format_one_status("AUTO", last_auto, auto_summary))
                if mode in ("manual", "compare"):
                    if top_roi is None or floor_roi is None:
                        lines.append("MANUAL: press m and select TOP, then FLOOR")
                    else:
                        lines.append(format_one_status("MANUAL", last_manual, manual_summary))
                if (
                    mode == "compare"
                    and auto_summary.stable
                    and manual_summary.stable
                ):
                    difference_cm = (
                        auto_summary.height_m - manual_summary.height_m
                    ) * 100.0
                    lines.append(f"AUTO - MANUAL: {difference_cm:+.2f} cm")
                if debug_reason and mode in ("auto", "compare"):
                    lines.append(f"AUTO status: {debug_reason}")
                lines.append(
                    "keys: 1 manual | 2 auto | 3 compare | m select ROI | "
                    "c clear | r reset | s save | q quit"
                )
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
                if key == ord("1"):
                    mode = "manual"
                    ready_announced = False
                    print("모드 변경: MANUAL")
                elif key == ord("2"):
                    mode = "auto"
                    ready_announced = False
                    print("모드 변경: AUTO")
                elif key == ord("3"):
                    mode = "compare"
                    ready_announced = False
                    print("모드 변경: COMPARE")
                elif key == ord("m"):
                    selected_top, selected_floor = select_manual_rois(color_image)
                    if selected_top is not None and selected_floor is not None:
                        top_roi, floor_roi = selected_top, selected_floor
                        manual_stabilizer.reset()
                        manual_summary = manual_stabilizer.summary()
                        last_manual = None
                        ready_announced = False
                        print(f"수동 ROI 설정 완료 | TOP={top_roi} | FLOOR={floor_roi}")
                elif key == ord("c"):
                    top_roi = None
                    floor_roi = None
                    last_manual = None
                    manual_stabilizer.reset()
                    manual_summary = manual_stabilizer.summary()
                    ready_announced = False
                    print("수동 ROI를 삭제했습니다.")
                elif key == ord("r"):
                    auto_stabilizer.reset()
                    manual_stabilizer.reset()
                    auto_summary = auto_stabilizer.summary()
                    manual_summary = manual_stabilizer.summary()
                    ready_announced = False
                    print("누적 측정값을 초기화했습니다.")
                elif key == ord("d"):
                    show_depth = not show_depth
                elif key == ord("s"):
                    entries = active_entries(
                        mode,
                        auto_summary,
                        manual_summary,
                        last_auto,
                        last_manual,
                    )
                    if not entries:
                        print("현재 모드에서 아직 STABLE인 결과가 없습니다.")
                    elif mode == "compare" and len(entries) < 2:
                        print("COMPARE 저장은 AUTO와 MANUAL 모두 STABLE이어야 합니다.")
                    else:
                        append_measurements_csv(args.csv, entries)
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

