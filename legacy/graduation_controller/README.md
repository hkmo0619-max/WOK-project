# Graduation Controller Legacy

이 디렉터리는 ROS2 전환 이전 졸업작품에서 사용한 Jetson 기반 통합 제어 코드를 보존하기 위한 공간입니다.

## 개발 흐름

1. Flask Web Controller + STM32 Serial 제어
2. YDLIDAR 기반 장애물 회피 기능 통합
3. Web + LiDAR + STM32 통합 구조 정리 및 최종 구현

현재 `controller.py`는 당시 보존된 최종 통합 버전입니다.

## 당시 제어 구조

Browser
→ Flask Web Controller
→ Jetson
→ STM32 Serial
→ STM32
→ Robot

LiDAR 데이터 처리와 장애물 회피 판단도 동일한 Python 프로그램 내부에서 수행했습니다.

당시 STM32 통신은 `/dev/stm32`, 115200 bps를 사용했으며,
주요 명령은 `PING`, `STOP`, `WHEEL`, `LEG`, `FWD`, `BACK`, `TL`, `TR` 형식이었습니다.

## 현재 ROS2 구현과의 관계

이 코드는 현재 ROS2 시스템의 실행 코드가 아닙니다.

현재 WoKi ROS2 구현은 저장소의 `src/` 디렉터리를 기준으로 하며,
과거 하나의 Python 프로그램에 통합되어 있던 기능을 LiDAR 처리, STM32 Bridge,
Web Controller, Control Manager 등의 ROS2 패키지로 분리하여 개발하고 있습니다.

## Git history note

이 디렉터리의 Git 이력은 2026년 5월에 보존된 개발 파일들을 기반으로
후에 정리한 것입니다. 따라서 Git commit 시각 자체가 당시 실제 개발 시각을
의미하지는 않습니다.
