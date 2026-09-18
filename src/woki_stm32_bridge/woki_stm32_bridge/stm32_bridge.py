#!/usr/bin/env python3
"""ROS2 transport bridge for the WoKi STM32 ASCII serial protocol."""

from __future__ import annotations

import threading
from typing import Optional

import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Bool, String


COMMAND_MAP = {
    "ping": "PING",
    "wheel": "WHEEL",
    "ramp": "RAMP",
    "leg": "LEG",
}

MOTION_COMMAND_MAP = {
    "wheel": {
        "forward": "WHEEL_FWD",
        "backward": "WHEEL_BACK",
        "left": "WHEEL_TL",
        "right": "WHEEL_TR",
        "stop": "WHEEL_STOP",
    },
    "leg": {
        "forward": "LEG_FWD",
        "backward": "LEG_BACK",
        "left": "LEG_LEFT",
        "right": "LEG_RIGHT",
        "stop": "LEG_STOP",
    },
}

ALLOWED_ROS_COMMANDS = {
    "forward",
    "backward",
    "left",
    "right",
    "stop",
    "wheel",
    "ramp",
    "leg",
}

STATUS_PUBLISH_PERIOD_SEC = 1.0
CONNECTION_CONTROL_PERIOD_SEC = 0.05
CONNECTION_STATES = {
    "DISCONNECTED",
    "CONNECTING",
    "HANDSHAKING",
    "CONNECTED",
}


class WokiStm32Bridge(Node):
    """Translate ROS2 semantic commands to the STM32 serial protocol."""

    def __init__(self) -> None:
        super().__init__("woki_stm32_bridge")

        self.declare_parameter("serial_port", "/dev/stm32")
        self.declare_parameter("baudrate", 460800)
        self.declare_parameter("serial_timeout_sec", 1.0)
        self.declare_parameter("stm32_heartbeat_period_sec", 1.0)
        self.declare_parameter("stm32_heartbeat_timeout_sec", 1.0)
        self.declare_parameter("stm32_reconnect_period_sec", 1.0)

        self.serial_port = str(self.get_parameter("serial_port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.serial_timeout_sec = float(
            self.get_parameter("serial_timeout_sec").value
        )
        self.stm32_heartbeat_period_sec = float(
            self.get_parameter("stm32_heartbeat_period_sec").value
        )
        self.stm32_heartbeat_timeout_sec = float(
            self.get_parameter("stm32_heartbeat_timeout_sec").value
        )
        self.stm32_reconnect_period_sec = float(
            self.get_parameter("stm32_reconnect_period_sec").value
        )

        self.serial_lock = threading.Lock()
        self.serial_connection: Optional[serial.Serial] = None
        self.connection_state = "DISCONNECTED"
        self.ready = False
        self.status = "DISCONNECTED"
        self.current_mode = "wheel"
        self.last_logged_command: Optional[str] = None
        self.last_reconnect_attempt_ns: Optional[int] = None
        self.ping_outstanding = False
        self.ping_started_ns: Optional[int] = None
        self.last_heartbeat_success_ns: Optional[int] = None
        self.receive_buffer = bytearray()
        self.last_disconnect_reason: Optional[str] = None

        self.ready_pub = self.create_publisher(
            Bool,
            "/woki/stm32/ready",
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            "/woki/stm32/status",
            10,
        )
        self.command_sub = self.create_subscription(
            String,
            "/woki/stm32/cmd",
            self.on_command,
            10,
        )
        self.status_timer = self.create_timer(
            STATUS_PUBLISH_PERIOD_SEC,
            self.publish_connection_state,
        )
        self.connection_timer = self.create_timer(
            CONNECTION_CONTROL_PERIOD_SEC,
            self.run_connection_state_machine,
        )

        self.publish_connection_state()

    def set_connection_state(self, state: str, status: str) -> None:
        if state not in CONNECTION_STATES:
            raise ValueError(f"Invalid STM32 connection state: {state}")

        ready = state == "CONNECTED" and status == "PING_OK"
        changed = (
            self.connection_state != state
            or self.ready != ready
            or self.status != status
        )
        self.connection_state = state
        self.ready = ready
        self.status = status
        self.publish_connection_state()

        if changed and self.context_is_ok():
            self.get_logger().info(
                f"STM32 status={self.status} | ready={self.ready}"
            )

    def context_is_ok(self) -> bool:
        return rclpy.ok(context=self.context)

    def publish_connection_state(self) -> None:
        if not self.context_is_ok():
            return

        self.ready_pub.publish(Bool(data=self.ready))
        self.status_pub.publish(String(data=self.status))

    def interval_expired(
        self,
        start_time_ns: Optional[int],
        interval_sec: float,
    ) -> bool:
        if start_time_ns is None:
            return True

        elapsed_ns = self.get_clock().now().nanoseconds - start_time_ns
        if elapsed_ns < 0:
            return True
        return elapsed_ns >= int(interval_sec * 1_000_000_000)

    def reset_ping_transaction(self) -> None:
        self.ping_outstanding = False
        self.ping_started_ns = None

    def serial_is_open(self) -> bool:
        try:
            return (
                self.serial_connection is not None
                and self.serial_connection.is_open
            )
        except Exception:
            return False

    def close_serial_transport(self) -> None:
        connection = self.serial_connection
        self.serial_connection = None
        if connection is None:
            return

        try:
            with self.serial_lock:
                if connection.is_open:
                    connection.close()
        except Exception:
            pass

    def mark_disconnected(self, reason: str) -> None:
        previous_state = self.connection_state
        self.close_serial_transport()
        self.receive_buffer.clear()
        self.reset_ping_transaction()
        self.last_heartbeat_success_ns = None
        self.last_reconnect_attempt_ns = self.get_clock().now().nanoseconds
        self.set_connection_state("DISCONNECTED", "DISCONNECTED")

        if (
            previous_state != "DISCONNECTED"
            and reason != self.last_disconnect_reason
            and self.context_is_ok()
        ):
            self.get_logger().warning(f"STM32 disconnected: {reason}")
        self.last_disconnect_reason = reason

    def attempt_serial_connection(self) -> None:
        self.last_reconnect_attempt_ns = self.get_clock().now().nanoseconds
        self.set_connection_state("CONNECTING", "CONNECTING")

        try:
            self.serial_connection = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=self.serial_timeout_sec,
                write_timeout=self.serial_timeout_sec,
            )
        except Exception as exc:
            self.serial_connection = None
            self.mark_disconnected(
                f"failed to open {self.serial_port}: {exc}"
            )
            return

        self.receive_buffer.clear()
        self.reset_ping_transaction()
        try:
            self.serial_connection.reset_input_buffer()
        except Exception as exc:
            self.mark_disconnected(f"input buffer reset failed: {exc}")
            return

        self.set_connection_state("HANDSHAKING", "HANDSHAKING")
        self.get_logger().info(
            f"Serial opened: {self.serial_port} @ {self.baudrate} bps"
        )
        self.start_ping_transaction()

    def write_serial_line(self, raw_command: str) -> bool:
        if not self.serial_is_open():
            self.mark_disconnected("serial port is not open")
            return False

        try:
            with self.serial_lock:
                self.serial_connection.write(
                    (raw_command + "\r\n").encode("ascii")
                )
                self.serial_connection.flush()
        except Exception as exc:
            self.mark_disconnected(f"serial write failed: {exc}")
            return False

        if self.context_is_ok():
            self.get_logger().debug(f"STM32 TX: {raw_command}")
        return True

    def start_ping_transaction(self) -> None:
        if self.ping_outstanding:
            return
        if not self.write_serial_line(COMMAND_MAP["ping"]):
            return

        self.ping_outstanding = True
        self.ping_started_ns = self.get_clock().now().nanoseconds

    def poll_serial_input(self) -> None:
        if not self.serial_is_open():
            self.mark_disconnected("serial port became unavailable")
            return

        try:
            with self.serial_lock:
                available = int(self.serial_connection.in_waiting)
                data = (
                    self.serial_connection.read(available)
                    if available > 0
                    else b""
                )
        except Exception as exc:
            self.mark_disconnected(f"serial read failed: {exc}")
            return

        if not data:
            return

        self.receive_buffer.extend(data)
        while b"\n" in self.receive_buffer:
            raw_line, _, remaining = self.receive_buffer.partition(b"\n")
            self.receive_buffer = bytearray(remaining)
            line = raw_line.decode("ascii", errors="ignore").strip()
            self.handle_serial_line(line)

        if len(self.receive_buffer) > 4096:
            self.receive_buffer.clear()

    def handle_serial_line(self, line: str) -> None:
        if self.context_is_ok():
            self.get_logger().debug(f"STM32 RX: {line}")
        if line != "OK" or not self.ping_outstanding:
            return

        self.reset_ping_transaction()
        self.last_heartbeat_success_ns = self.get_clock().now().nanoseconds
        if self.connection_state == "HANDSHAKING":
            self.last_disconnect_reason = None
            self.set_connection_state("CONNECTED", "PING_OK")

    def run_connection_state_machine(self) -> None:
        if self.connection_state == "DISCONNECTED":
            if self.interval_expired(
                self.last_reconnect_attempt_ns,
                self.stm32_reconnect_period_sec,
            ):
                self.attempt_serial_connection()
            return

        if self.connection_state == "CONNECTING":
            return

        if self.connection_state not in {"HANDSHAKING", "CONNECTED"}:
            self.mark_disconnected("invalid connection state")
            return

        self.poll_serial_input()
        if self.connection_state == "DISCONNECTED":
            return

        if self.ping_outstanding:
            if self.interval_expired(
                self.ping_started_ns,
                self.stm32_heartbeat_timeout_sec,
            ):
                self.mark_disconnected("PING response timeout")
            return

        if self.connection_state == "HANDSHAKING":
            self.start_ping_transaction()
            return


    def on_command(self, msg: String) -> None:
        command = str(msg.data).strip().lower()
        if command not in ALLOWED_ROS_COMMANDS:
            self.get_logger().warning(
                f"Ignored unsupported STM32 command: {command!r}"
            )
            return

        if self.connection_state != "CONNECTED" or not self.ready:
            self.get_logger().warning(
                f"Ignored STM32 command while not ready: {command}"
            )
            return

        self.send_protocol_command(command)

    def resolve_protocol_command(self, command: str) -> Optional[str]:
        raw_command = COMMAND_MAP.get(command)
        if raw_command is not None:
            return raw_command

        mode_commands = MOTION_COMMAND_MAP.get(self.current_mode)
        if mode_commands is None:
            return None
        return mode_commands.get(command)

    def send_protocol_command(self, command: str) -> bool:
        raw_command = self.resolve_protocol_command(command)
        if raw_command is None:
            if self.context_is_ok():
                self.get_logger().warning(
                    f"Ignored unknown protocol command: {command!r} "
                    f"(mode={self.current_mode!r})"
                )
            return False

        if self.connection_state != "CONNECTED" or not self.ready:
            return False

        if not self.write_serial_line(raw_command):
            return False

        if command in {"wheel", "leg"}:
            self.current_mode = command

        if (
            command != self.last_logged_command
            and self.context_is_ok()
        ):
            self.get_logger().info(f"TX → {raw_command}")
            self.last_logged_command = command
        return True

    def close_serial(self) -> None:
        if self.connection_state == "CONNECTED" and self.serial_is_open():
            self.send_protocol_command("stop")
        self.close_serial_transport()
        self.receive_buffer.clear()
        self.reset_ping_transaction()
        self.last_heartbeat_success_ns = None
        self.set_connection_state("DISCONNECTED", "DISCONNECTED")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WokiStm32Bridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_serial()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
