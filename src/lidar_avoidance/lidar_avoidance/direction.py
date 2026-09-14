import math
import time
import os
import glob
import atexit

try:
    import ydlidar
    YDLIDAR_OK = True
except Exception as e:
    ydlidar = None
    YDLIDAR_OK = False
    print("ydlidar import failed:", e)


# ============================================================
# LiDAR 기본 설정
# ============================================================
LIDAR_PORT = "auto"
LIDAR_BAUDRATE = 230400
LIDAR_SAMPLE_RATE = 5

# Check Sum 오류가 많이 뜨면 10.0보다 8.0이 더 안정적일 수 있음
LIDAR_SCAN_FREQ = 8.0

# 기존 졸업작품 코드에서는 0.28이었지만,
# 가까운 장애물이 None으로 빠질 수 있어서 테스트용으로 0.12로 낮춤
LIDAR_MIN_RANGE = 0.12
LIDAR_MAX_RANGE = 12.0

# 너무 가까운 이상값 제거용
TOO_CLOSE_DIST = 0.05

# 장애물 판단 거리
OBSTACLE_DIST = 0.50

# 한 섹터 안에 최소 몇 개의 점이 있어야 유효 거리로 볼지
# 1개만 잡히면 튀는 값일 수 있으므로 3개 이상으로 설정
MIN_POINTS_IN_SECTOR = 3


# ============================================================
# 방향 기준 설정
# ============================================================
# 기존 졸업작품 시연 코드 기준
# FRONT_CENTER_DEG = 180.0
#
# 만약 전방에 물체를 뒀는데 REAR가 줄어들면
# FRONT_CENTER_DEG = 0.0
# REAR_CENTER_DEG = 180.0
# 으로 바꾸면 됨.
FRONT_CENTER_DEG = 180.0
LEFT_CENTER_DEG = 90.0
RIGHT_CENTER_DEG = 270.0
REAR_CENTER_DEG = 0.0

# ============================================================
# 섹터 폭 설정
# ============================================================
# 현재 방향 기준 잡는 단계이므로 FRONT만 60도로 넓게 시작
# FRONT_WIDTH_DEG = 60.0 → 중심 기준 ±30도
#
# 방향 기준이 안정적으로 잡히면 최종 회피용으로
# FRONT_WIDTH_DEG = 45.0 으로 줄이면 됨.
FRONT_WIDTH_DEG = 60.0
LEFT_WIDTH_DEG = 45.0
RIGHT_WIDTH_DEG = 45.0
REAR_WIDTH_DEG = 45.0


laser = None


def find_lidar_ports():
    if LIDAR_PORT != "auto":
        return [LIDAR_PORT] if os.path.exists(LIDAR_PORT) else []

    ports = []

    if os.path.exists("/dev/ydlidar"):
        ports.append("/dev/ydlidar")

    for p in sorted(glob.glob("/dev/ttyUSB*")):
        if p not in ports:
            ports.append(p)

    return ports


def init_lidar():
    global laser

    if not YDLIDAR_OK:
        print("LiDAR library is not available.")
        return False

    ports = find_lidar_ports()

    if not ports:
        print("No LiDAR port found.")
        print("Check /dev/ydlidar or /dev/ttyUSB*")
        return False

    try:
        ydlidar.os_init()
    except Exception:
        pass

    for port in ports:
        print(f"Trying LiDAR port: {port}")

        try:
            L = ydlidar.CYdLidar()

            options = [
                ("LidarPropSerialPort", port),
                ("LidarPropSerialBaudrate", LIDAR_BAUDRATE),
                ("LidarPropLidarType", ydlidar.TYPE_TRIANGLE),
                ("LidarPropDeviceType", ydlidar.YDLIDAR_TYPE_SERIAL),
                ("LidarPropScanFrequency", LIDAR_SCAN_FREQ),
                ("LidarPropSampleRate", LIDAR_SAMPLE_RATE),
                ("LidarPropSingleChannel", False),
                ("LidarPropMaxAngle", 180.0),
                ("LidarPropMinAngle", -180.0),
                ("LidarPropMaxRange", LIDAR_MAX_RANGE),
                ("LidarPropMinRange", LIDAR_MIN_RANGE),
                ("LidarPropSupportMotorDtrCtrl", False),
            ]

            for name, val in options:
                if hasattr(ydlidar, name):
                    try:
                        L.setlidaropt(getattr(ydlidar, name), val)
                    except Exception:
                        pass

            # 기존 코드의 Intensity 옵션 오타/정상명 둘 다 대응
            if hasattr(ydlidar, "LidarPropIntenstiy"):
                try:
                    L.setlidaropt(ydlidar.LidarPropIntenstiy, True)
                except Exception:
                    pass
            elif hasattr(ydlidar, "LidarPropIntensity"):
                try:
                    L.setlidaropt(ydlidar.LidarPropIntensity, True)
                except Exception:
                    pass

            if not L.initialize():
                print(f"Initialize failed: {port}")
                L.disconnecting()
                continue

            if not L.turnOn():
                print(f"Turn on failed: {port}")
                L.turnOff()
                L.disconnecting()
                continue

            laser = L
            print(f"LiDAR OK: {port}")
            return True

        except Exception as e:
            print(f"LiDAR error on {port}: {e}")
            try:
                L.disconnecting()
            except Exception:
                pass

    print("LiDAR start failed.")
    return False


def close_lidar():
    global laser

    try:
        if laser:
            laser.turnOff()
            laser.disconnecting()
            laser = None
            print("LiDAR closed.")
    except Exception as e:
        print("LiDAR close error:", e)


def angle_diff_deg(deg, center_deg):
    # 사진 속 코드와 같은 원리
    # abs((deg - center + 180) % 360 - 180)
    return abs((deg - center_deg + 180.0) % 360.0 - 180.0)


def get_sector_info(scan, center_deg, width_deg):
    dists = []

    for p in scan.points:
        deg = math.degrees(p.angle) % 360.0
        d = p.range

        if math.isinf(d) or math.isnan(d):
            continue

        if d <= TOO_CLOSE_DIST:
            continue

        if not (LIDAR_MIN_RANGE <= d <= LIDAR_MAX_RANGE):
            continue

        # 사진 속 계산식과 같은 방식
        if angle_diff_deg(deg, center_deg) <= width_deg / 2.0:
            dists.append(d)

    count = len(dists)

    # 점이 너무 적으면 그 방향에 확실한 물체가 있다고 보지 않음
    if count < MIN_POINTS_IN_SECTOR:
        return None, count

    return min(dists), count


def read_directions():
    if laser is None:
        return None

    scan = ydlidar.LaserScan()

    if not laser.doProcessSimple(scan):
        return None

    front, front_n = get_sector_info(scan, FRONT_CENTER_DEG, FRONT_WIDTH_DEG)
    left, left_n = get_sector_info(scan, LEFT_CENTER_DEG, LEFT_WIDTH_DEG)
    right, right_n = get_sector_info(scan, RIGHT_CENTER_DEG, RIGHT_WIDTH_DEG)
    rear, rear_n = get_sector_info(scan, REAR_CENTER_DEG, REAR_WIDTH_DEG)

    return {
        "front": front,
        "left": left,
        "right": right,
        "rear": rear,
        "front_n": front_n,
        "left_n": left_n,
        "right_n": right_n,
        "rear_n": rear_n,
    }


def fmt_dist(d):
    return "None" if d is None else f"{d:.2f}m"


def obstacle_state(d):
    if d is None:
        return "NO_DATA"
    if d <= OBSTACLE_DIST:
        return "BLOCKED"
    return "CLEAR"


def choose_avoid_direction(data):
    front = data["front"]
    left = data["left"]
    right = data["right"]
    rear = data["rear"]

    # 전방 데이터가 없으면 일단 정지 판단
    if front is None:
        return "STOP_CHECK_FRONT"

    # 전방이 뚫려 있으면 직진
    if front > OBSTACLE_DIST:
        return "FORWARD"

    # 여기부터는 전방 장애물 감지 상태
    left_clear = left is None or left > OBSTACLE_DIST
    right_clear = right is None or right > OBSTACLE_DIST

    if right_clear and not left_clear:
        return "TANK_RIGHT"

    if left_clear and not right_clear:
        return "TANK_LEFT"

    if right_clear and left_clear:
        # 둘 다 가능하면 기존 졸업작품 시연처럼 우회전 우선
        return "TANK_RIGHT"

    # 좌우가 둘 다 막혔고 후방이 가능하면 후진 후보
    if rear is None or rear > OBSTACLE_DIST:
        return "BACKWARD"

    return "STOP"


def print_settings():
    print()
    print("========== LiDAR Direction Settings ==========")
    print(f"FRONT_CENTER_DEG = {FRONT_CENTER_DEG}")
    print(f"LEFT_CENTER_DEG  = {LEFT_CENTER_DEG}")
    print(f"RIGHT_CENTER_DEG = {RIGHT_CENTER_DEG}")
    print(f"REAR_CENTER_DEG  = {REAR_CENTER_DEG}")
    print()
    print(f"FRONT_WIDTH_DEG = {FRONT_WIDTH_DEG} deg, center ±{FRONT_WIDTH_DEG / 2.0} deg")
    print(f"LEFT_WIDTH_DEG  = {LEFT_WIDTH_DEG} deg, center ±{LEFT_WIDTH_DEG / 2.0} deg")
    print(f"RIGHT_WIDTH_DEG = {RIGHT_WIDTH_DEG} deg, center ±{RIGHT_WIDTH_DEG / 2.0} deg")
    print(f"REAR_WIDTH_DEG  = {REAR_WIDTH_DEG} deg, center ±{REAR_WIDTH_DEG / 2.0} deg")
    print()
    print(f"LIDAR_MIN_RANGE = {LIDAR_MIN_RANGE} m")
    print(f"LIDAR_MAX_RANGE = {LIDAR_MAX_RANGE} m")
    print(f"OBSTACLE_DIST   = {OBSTACLE_DIST} m")
    print(f"MIN_POINTS_IN_SECTOR = {MIN_POINTS_IN_SECTOR}")
    print("==============================================")
    print()


def main():
    if not init_lidar():
        return

    print_settings()
    print("LiDAR direction recognition started.")
    print("Press Ctrl+C to stop.")
    print()

    try:
        while True:
            data = read_directions()

            if data is None:
                print("No LiDAR data")
                time.sleep(0.1)
                continue

            front = data["front"]
            left = data["left"]
            right = data["right"]
            rear = data["rear"]

            decision = choose_avoid_direction(data)

            print(
                f"FRONT={fmt_dist(front):>7} [{obstacle_state(front):>7}, n={data['front_n']:>2}] | "
                f"LEFT={fmt_dist(left):>7} [{obstacle_state(left):>7}, n={data['left_n']:>2}] | "
                f"RIGHT={fmt_dist(right):>7} [{obstacle_state(right):>7}, n={data['right_n']:>2}] | "
                f"REAR={fmt_dist(rear):>7} [{obstacle_state(rear):>7}, n={data['rear_n']:>2}] | "
                f"DECISION={decision}"
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        close_lidar()


atexit.register(close_lidar)


if __name__ == "__main__":
    main()
