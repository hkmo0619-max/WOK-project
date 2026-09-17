from flask import Flask, request, render_template_string
import serial
import time
import threading

app = Flask(__name__)

SERIAL_PORT = "/dev/stm32"
BAUD_RATE = 115200

COMMAND_MAP = {
    "forward": "F",
    "backward": "B",
    "left": "L",
    "right": "R",
    "stop": "S",
    "wheel_mode": "WHEEL",
    "leg_mode": "LEG",
    "tank_right": "TR"
}

ser = None
current_command = "stop"
last_command_time = time.time()


def connect_serial():
    global ser

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[OK] STM32 connected: {SERIAL_PORT}, baud={BAUD_RATE}")
    except Exception as e:
        ser = None
        print(f"[WARN] STM32 serial open failed: {e}")
        print("[WARN] Web controller will run, but commands will not be sent to STM32.")


def send_to_stm32(cmd):
    global current_command, last_command_time

    if cmd not in COMMAND_MAP:
        print(f"[ERROR] Invalid command: {cmd}")
        return False

    stm32_cmd = COMMAND_MAP[cmd]
    message = stm32_cmd + "\n"

    current_command = cmd
    last_command_time = time.time()

    print(f"[CMD] {cmd} -> STM32: {stm32_cmd}")

    if ser is not None:
        try:
            ser.write(message.encode())
        except Exception as e:
            print(f"[ERROR] Serial write failed: {e}")
            return False

    return True


def safety_watchdog():
    global current_command

    while True:
        if current_command != "stop":
            elapsed = time.time() - last_command_time

            if elapsed > 0.7:
                print("[SAFETY] command timeout -> STOP")

                if ser is not None:
                    try:
                        ser.write(b"S\n")
                    except Exception as e:
                        print(f"[ERROR] Safety stop failed: {e}")

                current_command = "stop"

        time.sleep(0.1)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Robot Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {
            background: #111;
            color: white;
            text-align: center;
            font-family: Arial;
            padding-top: 40px;
            user-select: none;
        }

        h1 {
            font-size: 30px;
            margin-bottom: 30px;
        }

        button {
            width: 120px;
            height: 80px;
            font-size: 24px;
            margin: 8px;
            border-radius: 15px;
            border: none;
        }

        button:active {
            background: #4da3ff;
        }

        .stop {
            background: red;
            color: white;
        }

        .mode {
            width: 130px;
            height: 75px;
            font-size: 22px;
        }
    </style>
</head>

<body>
    <h1>Robot Controller</h1>

    <div>
        <button id="forward">↑</button>
    </div>

    <div>
        <button id="left">←</button>
        <button class="stop" onclick="sendCmd('stop')">STOP</button>
        <button id="right">→</button>
    </div>

    <div>
        <button id="backward">↓</button>
    </div>

    <br>

    <button class="mode" onclick="sendCmd('wheel_mode')">Wheel</button>
    <button class="mode" onclick="sendCmd('leg_mode')">Leg</button>
    <button class="mode" onclick="sendCmd('tank_right')">Tank R</button>

    <script>
        function sendCmd(cmd) {
            fetch('/cmd', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({cmd: cmd})
            });
        }

        function bindHoldButton(id, cmd) {
            const btn = document.getElementById(id);

            btn.addEventListener("mousedown", function(e) {
                e.preventDefault();
                sendCmd(cmd);
            });

            btn.addEventListener("mouseup", function(e) {
                e.preventDefault();
                sendCmd("stop");
            });

            btn.addEventListener("mouseleave", function(e) {
                e.preventDefault();
                sendCmd("stop");
            });

            btn.addEventListener("touchstart", function(e) {
                e.preventDefault();
                sendCmd(cmd);
            });

            btn.addEventListener("touchend", function(e) {
                e.preventDefault();
                sendCmd("stop");
            });
        }

        bindHoldButton("forward", "forward");
        bindHoldButton("backward", "backward");
        bindHoldButton("left", "left");
        bindHoldButton("right", "right");
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/cmd", methods=["POST"])
def cmd():
    data = request.get_json()
    command = data.get("cmd")

    ok = send_to_stm32(command)

    if ok:
        return {"status": "ok", "cmd": command}
    else:
        return {"status": "error", "cmd": command}, 400


if __name__ == "__main__":
    connect_serial()

    watchdog_thread = threading.Thread(target=safety_watchdog, daemon=True)
    watchdog_thread.start()

    app.run(host="0.0.0.0", port=5000)
