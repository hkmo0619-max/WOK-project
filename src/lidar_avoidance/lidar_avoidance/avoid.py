import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String


# 방향별 중심각과 감지 폭
SECTORS = {
    "front": (180.0, 60.0),
    "left":  (120.0, 60.0),
    "right": (240.0, 60.0),
    "rear":  (0.0, 60.0),
}

OBSTACLE_DIST = 0.60
MIN_RANGE = 0.12
MAX_RANGE = 12.0

PRINT_CMD_LOG = True
PRINT_STATUS_LOG = True
STATUS_LOG_INTERVAL = 1.0

CMD_LABELS = {
    "forward": "FWD",
    "tank_left": "TL",
    "tank_right": "TR",
    "stop": "STOP",
}


def angle_diff(a, b):
    """두 각도의 최소 차이를 0~180도 범위로 반환"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def format_distance(distance):
    return "None" if distance is None else f"{distance:.2f}m"


class LidarAvoidanceNode(Node):
    def __init__(self):
        super().__init__("lidar_avoidance_ros2")

        self.declare_parameter("direct_cmd_enabled", True)
        self.direct_cmd_enabled = self.get_parameter(
            "direct_cmd_enabled"
        ).value

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            scan_qos,
        )

        self.cmd_pub = None
        if self.direct_cmd_enabled:
            self.cmd_pub = self.create_publisher(
                String,
                "/robot/cmd",
                10,
            )
        self.front_pub = self.create_publisher(
            Float32,
            "/lidar_avoidance/front_m",
            10,
        )
        self.left_pub = self.create_publisher(
            Float32,
            "/lidar_avoidance/left_m",
            10,
        )
        self.right_pub = self.create_publisher(
            Float32,
            "/lidar_avoidance/right_m",
            10,
        )
        self.rear_pub = self.create_publisher(
            Float32,
            "/lidar_avoidance/rear_m",
            10,
        )
        self.suggested_cmd_pub = self.create_publisher(
            String,
            "/lidar_avoidance/suggested_cmd",
            10,
        )

        self.last_cmd = None
        self.last_status_time = 0.0

        self.get_logger().info(
            f"LiDAR avoidance started | "
            f"distance={OBSTACLE_DIST:.2f}m | "
            f"front_width={SECTORS['front'][1]:.0f}deg | "
            f"direct_cmd={'ON' if self.direct_cmd_enabled else 'OFF'}"
        )

    def get_sector_distance(self, scan, center, width):
        """지정한 섹터에서 가장 가까운 거리 반환"""
        min_distance = None

        for index, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue

            if not MIN_RANGE <= distance <= MAX_RANGE:
                continue

            angle = math.degrees(
                scan.angle_min + index * scan.angle_increment
            ) % 360.0

            if angle_diff(angle, center) <= width / 2.0:
                if min_distance is None or distance < min_distance:
                    min_distance = distance

        return min_distance

    def publish_cmd(self, cmd):
        """명령이 변경됐을 때만 발행"""
        if not self.direct_cmd_enabled or self.cmd_pub is None:
            return

        if cmd == self.last_cmd:
            return

        self.cmd_pub.publish(String(data=cmd))
        self.last_cmd = cmd

        if PRINT_CMD_LOG:
            self.get_logger().info(
                f"CMD → {CMD_LABELS.get(cmd, cmd.upper())}"
            )

    def publish_distance(self, publisher, distance):
        """거리 데이터가 없으면 NaN으로 발행"""
        value = float("nan") if distance is None else float(distance)
        publisher.publish(Float32(data=value))

    def publish_control_status(self, distances, cmd):
        """Control Manager가 사용할 거리와 추천 행동을 발행"""
        self.publish_distance(self.front_pub, distances["front"])
        self.publish_distance(self.left_pub, distances["left"])
        self.publish_distance(self.right_pub, distances["right"])
        self.publish_distance(self.rear_pub, distances["rear"])
        self.suggested_cmd_pub.publish(String(data=cmd))

    def decide_cmd(self, front, left, right):
        """전방 장애물 감지 후 회피 방향 결정"""
        if front is None:
            return "stop"

        if front > OBSTACLE_DIST:
            return "forward"

        if left is not None and right is not None:
            return "tank_right" if right >= left else "tank_left"

        if right is not None:
            return "tank_right"

        if left is not None:
            return "tank_left"

        return "stop"

    def print_status(self, distances, cmd):
        """1초마다 거리와 현재 명령 출력"""
        if not PRINT_STATUS_LOG:
            return

        now = self.get_clock().now().nanoseconds / 1e9

        if now - self.last_status_time < STATUS_LOG_INTERVAL:
            return

        self.get_logger().info(
            f"FRONT={format_distance(distances['front'])} | "
            f"LEFT={format_distance(distances['left'])} | "
            f"RIGHT={format_distance(distances['right'])} | "
            f"REAR={format_distance(distances['rear'])} | "
            f"CMD={CMD_LABELS.get(cmd, cmd.upper())}"
        )

        self.last_status_time = now

    def scan_callback(self, scan):
        distances = {
            name: self.get_sector_distance(scan, center, width)
            for name, (center, width) in SECTORS.items()
        }

        cmd = self.decide_cmd(
            distances["front"],
            distances["left"],
            distances["right"],
        )

        self.publish_cmd(cmd)
        self.publish_control_status(distances, cmd)
        self.print_status(distances, cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LidarAvoidanceNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        # ROS 컨텍스트가 살아 있을 때만 STOP 발행
        if rclpy.ok():
            node.publish_cmd("stop")
            node.get_logger().info("Node stopped by Ctrl+C")

    finally:
        node.destroy_node()

        # 이미 종료됐더라도 오류가 발생하지 않도록 처리
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
