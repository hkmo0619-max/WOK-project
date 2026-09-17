from flask import Flask, render_template_string, jsonify
import serial
import time
import threading
import atexit
import math
import os
import glob
import logging

logging.getLogger("werkzeug").setLevel(logging.ERROR)

try:
    import ydlidar
    YDLIDAR_AVAILABLE = True
except ImportError:
    ydlidar = None
    YDLIDAR_AVAILABLE = False

SERIAL_PORT = "/dev/stm32"
BAUDRATE    = 115200
LIDAR_PORT  = "auto"

LIDAR_BAUDRATE    = 230400
LIDAR_SAMPLE_RATE = 5
LIDAR_SCAN_FREQ   = 10.0
LIDAR_MIN_RANGE   = 0.28
LIDAR_MAX_RANGE   = 16.0

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000

HOLD_REPEAT_MS   = 100
STATUS_UPDATE_MS = 200
MODE_SWITCH_TIME = 1.0

OBSTACLE_DISTANCE_M  = 0.50
TOO_CLOSE_DISTANCE_M = 0.12
AVOID_FORWARD_TIME   = 0.08
AVOID_LOOP_DELAY     = 0.03
AVOID_CMD_REPEAT_TIME = 0.10

FRONT_CENTER_DEG = 180.0
FRONT_WIDTH_DEG  = 30.0

STM32_CMD_MAP = {
    "ping": "PING", "stop": "STOP",
    "wheel": "WHEEL", "leg": "LEG",
    "forward": "FWD", "backward": "BACK",
    "tank_left": "TL", "tank_right": "TR",
}
DRIVE_COMMANDS = {"forward", "backward", "tank_left", "tank_right", "stop"}

app = Flask(__name__)

ser         = None
serial_lock = threading.Lock()
laser       = None
lidar_lock  = threading.Lock()

avoid_thread     = None
avoid_stop_event = threading.Event()

state_lock = threading.Lock()
state = {
    "mode": "wheel", "motion": "stop",
    "mode_switching": False, "auto_avoid": False,
    "front_distance": None, "message": "Ready",
}

def log_info(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def log_tx(cmd):   print(f"[{time.strftime('%H:%M:%S')}] TX → {cmd}", flush=True)

def set_state(**kw):
    with state_lock: state.update(kw)

def get_state_copy():
    with state_lock: return dict(state)

# ── Serial ────────────────────────────────────────────────────────────────────
def init_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        time.sleep(2.0)
        log_info(f"[OK] Serial: {SERIAL_PORT}")
        set_state(message=f"Serial opened: {SERIAL_PORT}")
        return True
    except Exception as e:
        log_info(f"[ERR] Serial: {e}"); set_state(message="Serial open failed"); ser = None; return False

def send_to_stm32(cmd, wait_reply=False):
    global ser
    cmd = cmd.lower()
    if cmd not in STM32_CMD_MAP: log_info(f"[ERR] Unknown: {cmd}"); return None
    if ser is None or not ser.is_open: log_info("[ERR] Serial not open"); return None
    stm32_cmd = STM32_CMD_MAP[cmd]
    try:
        with serial_lock:
            if wait_reply: ser.reset_input_buffer()
            ser.write((stm32_cmd + "\r\n").encode("ascii")); ser.flush(); log_tx(stm32_cmd)
            if wait_reply:
                resp = ser.readline().decode("ascii", errors="ignore").strip()
                log_info(f"RX ← {resp}"); return resp
    except Exception as e:
        log_info(f"[ERR] Send: {e}")
    return None

def ping_stm32():
    resp = send_to_stm32("ping", wait_reply=True)
    if resp == "OK":
        log_info("[OK] STM32 connected"); set_state(message="STM32 connected: OK"); return True
    log_info("[WARN] STM32 PING failed"); set_state(message="STM32 PING failed"); return False

# ── LiDAR ─────────────────────────────────────────────────────────────────────
def find_lidar_ports():
    if LIDAR_PORT != "auto":
        return [LIDAR_PORT] if os.path.exists(LIDAR_PORT) else []
    ports = []
    if os.path.exists("/dev/ydlidar"): ports.append("/dev/ydlidar")
    for p in sorted(glob.glob("/dev/ttyUSB*")):
        if p not in ports: ports.append(p)
    return ports

def set_lidar_opt(obj, name, value):
    if hasattr(ydlidar, name):
        try: obj.setlidaropt(getattr(ydlidar, name), value); return True
        except Exception as e: log_info(f"[WARN] set {name}: {e}")
    return False

def init_lidar():
    global laser
    if not YDLIDAR_AVAILABLE: set_state(message="LiDAR not available"); return False
    ports = find_lidar_ports()
    if not ports: log_info("[ERR] No LiDAR port"); set_state(message="No LiDAR port"); return False
    try: ydlidar.os_init()
    except Exception: pass

    for port in ports:
        try:
            ldr = ydlidar.CYdLidar()
            ldr.setlidaropt(ydlidar.LidarPropSerialPort,     port)
            ldr.setlidaropt(ydlidar.LidarPropSerialBaudrate, LIDAR_BAUDRATE)
            ldr.setlidaropt(ydlidar.LidarPropLidarType,      ydlidar.TYPE_TRIANGLE)
            ldr.setlidaropt(ydlidar.LidarPropDeviceType,     ydlidar.YDLIDAR_TYPE_SERIAL)
            ldr.setlidaropt(ydlidar.LidarPropScanFrequency,  LIDAR_SCAN_FREQ)
            ldr.setlidaropt(ydlidar.LidarPropSampleRate,     LIDAR_SAMPLE_RATE)
            ldr.setlidaropt(ydlidar.LidarPropSingleChannel,  False)
            ldr.setlidaropt(ydlidar.LidarPropMaxAngle,       180.0)
            ldr.setlidaropt(ydlidar.LidarPropMinAngle,      -180.0)
            ldr.setlidaropt(ydlidar.LidarPropMaxRange,       LIDAR_MAX_RANGE)
            ldr.setlidaropt(ydlidar.LidarPropMinRange,       LIDAR_MIN_RANGE)
            if not set_lidar_opt(ldr, "LidarPropIntenstiy", True):
                set_lidar_opt(ldr, "LidarPropIntensity", True)
            set_lidar_opt(ldr, "LidarPropSupportMotorDtrCtrl", False)

            if not ldr.initialize(): ldr.disconnecting(); continue
            if not ldr.turnOn(): ldr.turnOff(); ldr.disconnecting(); continue

            laser = ldr; log_info(f"[OK] LiDAR: {port}"); set_state(message=f"LiDAR: {port}"); return True
        except Exception as e:
            log_info(f"[WARN] LiDAR {port}: {e}")
            try: ldr.disconnecting()
            except Exception: pass

    log_info("[ERR] LiDAR start failed"); set_state(message="LiDAR start failed"); return False

def close_lidar():
    global laser
    try:
        if laser: laser.turnOff(); laser.disconnecting(); laser = None
    except Exception as e: log_info(f"[WARN] LiDAR close: {e}")

def get_front_distance():
    if laser is None: return None
    try:
        scan = ydlidar.LaserScan()
        with lidar_lock:
            if not laser.doProcessSimple(scan): return None
        dists = []
        for p in scan.points:
            deg = math.degrees(p.angle)
            if deg < 0: deg += 360.0
            d = p.range
            if d <= TOO_CLOSE_DISTANCE_M or d <= 0.0: continue
            if not (LIDAR_MIN_RANGE <= d <= LIDAR_MAX_RANGE): continue
            if abs((deg - FRONT_CENTER_DEG + 180.0) % 360.0 - 180.0) <= FRONT_WIDTH_DEG / 2.0:
                dists.append(d)
        return min(dists) if dists else None
    except Exception as e:
        log_info(f"[WARN] LiDAR scan: {e}"); return None

# ── 모드 전환 ─────────────────────────────────────────────────────────────────
def finish_mode_switch(target):
    time.sleep(MODE_SWITCH_TIME)
    set_state(mode=target, mode_switching=False, motion="stop", message=f"Mode: {target}")
    log_info(f"[MODE] → {target}")

def start_mode_switch(target):
    target = target.lower()
    if target not in ("wheel", "leg"): return False, "Unknown mode"
    cur = get_state_copy()
    if cur["mode_switching"]: return False, "Mode switching in progress"
    if target == cur["mode"]: return True, f"Already {target}"
    set_state(mode_switching=True, auto_avoid=False, motion="stop", message=f"Switching to {target}")
    stop_obstacle_avoidance(send_stop=False, update_message=False)
    send_to_stm32("stop"); time.sleep(0.1); send_to_stm32(target)
    threading.Thread(target=finish_mode_switch, args=(target,), daemon=True).start()
    return True, f"Switching to {target}"

# ── 장애물 회피 ───────────────────────────────────────────────────────────────
def _turn_and_check():
    """TR 후 전방 재확인. 최대 3회 추가 회전. 확보되면 True 반환."""
    send_to_stm32("stop"); set_state(motion="stop"); time.sleep(0.10)
    for _ in range(3):
        if avoid_stop_event.is_set(): return False
        chk = get_front_distance(); set_state(front_distance=chk)
        if chk is None or chk > OBSTACLE_DISTANCE_M: return True
        log_info(f"[AUTO] 여전히 막힘: {chk:.2f} m")
        send_to_stm32("tank_right"); set_state(motion="tank_right"); time.sleep(AVOID_TURN_TIME)
        send_to_stm32("stop"); set_state(motion="stop"); time.sleep(0.10)
    return not avoid_stop_event.is_set()

def obstacle_avoidance_loop():
    log_info("[AUTO] 회피 시작")
    set_state(auto_avoid=True, motion="stop", message="Obstacle avoidance started")

    last_cmd = None
    last_tx_time = 0.0

    def send_auto_cmd(cmd, message):
        """
        같은 명령을 너무 과하게 보내지 않도록 일정 주기로만 재전송.
        단, 명령이 바뀌면 즉시 전송.
        """
        nonlocal last_cmd, last_tx_time

        now = time.time()

        if cmd != last_cmd or (now - last_tx_time) >= AVOID_CMD_REPEAT_TIME:
            send_to_stm32(cmd)
            last_cmd = cmd
            last_tx_time = now

        set_state(motion=cmd, message=message)

    while not avoid_stop_event.is_set():
        cur = get_state_copy()

        if cur["mode"] != "wheel":
            send_to_stm32("stop")
            set_state(motion="stop", message="Auto avoid: not wheel mode")
            break

        if cur["mode_switching"]:
            send_to_stm32("stop")
            set_state(motion="stop", message="Auto avoid: mode switching")
            break

        dist = get_front_distance()
        set_state(front_distance=dist)

        if dist is None:
            send_auto_cmd("stop", "No LiDAR data")
            time.sleep(AVOID_LOOP_DELAY)
            continue

        if dist <= OBSTACLE_DISTANCE_M:
            # 장애물이 0.5m 이내면 STOP을 반복하지 않고 TR을 계속 유지
            log_info(f"[AUTO] 장애물 감지: {dist:.2f} m -> TR 유지")
            send_auto_cmd("tank_right", f"Avoiding TR: {dist:.2f} m")

        else:
            # 전방이 0.5m 이상 확보되면 바로 FWD로 전환
            send_auto_cmd("forward", f"Forward: {dist:.2f} m")

        time.sleep(AVOID_LOOP_DELAY)

    send_to_stm32("stop")
    set_state(auto_avoid=False, motion="stop", message="Obstacle avoidance stopped")
    log_info("[AUTO] 회피 종료")

def start_obstacle_avoidance():
    global avoid_thread
    if laser is None: return False, "LiDAR not ready"
    cur = get_state_copy()
    if cur["mode_switching"]: return False, "Mode switching in progress"
    if cur["mode"] != "wheel": return False, "Wheel mode only"
    if cur["auto_avoid"]: return True, "Already running"
    avoid_stop_event.clear()
    avoid_thread = threading.Thread(target=obstacle_avoidance_loop, daemon=True)
    avoid_thread.start()
    return True, "Obstacle avoidance started"

def stop_obstacle_avoidance(send_stop=True, update_message=True):
    global avoid_thread
    avoid_stop_event.set(); avoid_thread = None
    if send_stop: send_to_stm32("stop")
    if update_message: set_state(auto_avoid=False, motion="stop", message="Obstacle avoidance stopped")
    else: set_state(auto_avoid=False, motion="stop")

# ── Web UI ────────────────────────────────────────────────────────────────────
html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Robot Controller</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 35px;
               user-select: none; touch-action: none; background: #f4f4f4; }
        h1 { margin-bottom: 20px; }
        .status { display: inline-block; min-width: 360px; padding: 15px; margin-bottom: 20px;
                  background: white; border-radius: 12px; border: 1px solid #ccc;
                  font-size: 18px; line-height: 1.7; }
        button { width: 150px; height: 75px; font-size: 22px; margin: 8px;
                 border-radius: 14px; border: 1px solid #555; touch-action: none; cursor: pointer; }
        button:active { background-color: #cccccc; }
        button:disabled { opacity: 0.45; cursor: not-allowed; }
        .move { background-color: #eeeeee; }
        .stop { background-color: red; color: white; font-weight: bold; }
        .mode { background-color: #ddeeff; }
        .auto-on  { background-color: #ddffdd; }
        .auto-off { background-color: #ffe0cc; }
        .small { font-size: 15px; color: #555; margin-top: 15px; }
    </style>
</head>
<body>
    <h1>Robot Web Controller</h1>
    <div class="status">
        <div>Mode: <b id="mode">-</b></div>
        <div>Motion: <b id="motion">-</b></div>
        <div>Mode Switching: <b id="mode_switching">-</b></div>
        <div>Auto Avoid: <b id="auto_avoid">-</b></div>
        <div>Front Distance: <b id="front_distance">-</b></div>
        <div>Message: <b id="message">-</b></div>
    </div>
    <div>
        <button class="mode" onclick="setMode('wheel')">WHEEL</button>
        <button class="mode" onclick="setMode('leg')">LEG</button>
    </div>
    <div><button class="move drive-btn" id="forward">Forward</button></div>
    <div>
        <button class="move drive-btn" id="tank_left">Tank Left</button>
        <button class="stop" onclick="stopAll()">Stop</button>
        <button class="move drive-btn" id="tank_right">Tank Right</button>
    </div>
    <div><button class="move drive-btn" id="backward">Backward</button></div>
    <div style="margin-top:20px;">
        <button class="auto-on" id="avoid_on" onclick="startAvoid()">Avoid ON</button>
        <button class="auto-off" onclick="stopAvoid()">Avoid OFF</button>
    </div>
    <p class="small">
        전진 / 후진 / 회전은 wheel mode에서만 작동합니다.<br>
        버튼을 누르고 있으면 명령이 반복 전송되고, 떼면 정지됩니다.<br>
        장애물 회피는 wheel mode에서만 작동합니다.
    </p>
<script>
"use strict";
let activeCmd = null, repeatTimer = null, stoppingInProgress = false;
const REPEAT_MS = __HOLD_REPEAT_MS__, STATUS_MS = __STATUS_UPDATE_MS__;

const post = url => fetch(url, { method: "POST" }).then(r => r.json()).then(d => { updateStatus(); return d; });
const sendCmd = (cmd, refresh) => fetch("/cmd/" + cmd, { method: "POST" })
    .then(r => r.json()).then(d => { if (refresh !== false) updateStatus(); return d; });

const setMode    = m => { stopRepeat(); sendStop(); post("/mode/" + m); };
const startAvoid = () => { stopRepeat(); post("/avoid/start"); };
const stopAvoid  = () => { stopRepeat(); post("/avoid/stop"); };
const stopAll    = () => { stopRepeat(); sendStop(); };

function stopRepeat() {
    if (repeatTimer) { clearInterval(repeatTimer); repeatTimer = null; }
    activeCmd = null;
}

function sendStop() {
    if (stoppingInProgress) return;
    stoppingInProgress = true;
    fetch("/cmd/stop", { method: "POST", keepalive: true })
        .then(r => r.json()).then(() => updateStatus())
        .catch(e => console.error("stop:", e))
        .finally(() => { stoppingInProgress = false; });
}

function startCommand(cmd, e) {
    e.preventDefault(); e.stopPropagation();
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch (_) {}
    const btn = document.getElementById(cmd);
    if (btn && btn.disabled) return;
    stopRepeat(); activeCmd = cmd; stoppingInProgress = false;
    sendCmd(cmd, true);
    repeatTimer = setInterval(() => { if (activeCmd) sendCmd(activeCmd, false); }, REPEAT_MS);
}

function stopCommand(e) {
    e.preventDefault(); e.stopPropagation();
    if (!activeCmd) return;
    stopRepeat(); sendStop();
}

function setupHoldButton(id, cmd) {
    const btn = document.getElementById(id);
    btn.addEventListener("pointerdown",        e => startCommand(cmd, e));
    btn.addEventListener("pointerup",          stopCommand);
    btn.addEventListener("pointercancel",      stopCommand);
    btn.addEventListener("lostpointercapture", stopCommand);
    btn.addEventListener("pointerleave",       e => { if (activeCmd === cmd) stopCommand(e); });
    btn.addEventListener("contextmenu",        e => e.preventDefault());
}

setupHoldButton("forward",    "forward");
setupHoldButton("backward",   "backward");
setupHoldButton("tank_left",  "tank_left");
setupHoldButton("tank_right", "tank_right");

window.addEventListener("blur", () => { if (activeCmd) { stopRepeat(); sendStop(); } });
document.addEventListener("visibilitychange", () => { if (document.hidden && activeCmd) { stopRepeat(); sendStop(); } });
document.addEventListener("contextmenu", e => e.preventDefault());

function updateStatus() {
    fetch("/status").then(r => r.json()).then(d => {
        ["mode","motion","mode_switching","auto_avoid","message"].forEach(k =>
            document.getElementById(k).innerText = d[k]);
        document.getElementById("front_distance").innerText =
            d.front_distance === null ? "None" : Number(d.front_distance).toFixed(2) + " m";
        const blocked = d.mode_switching || d.mode !== "wheel" || d.auto_avoid;
        document.querySelectorAll(".drive-btn").forEach(b => b.disabled = blocked);
        document.getElementById("avoid_on").disabled = d.mode_switching || d.mode !== "wheel";
    }).catch(e => console.error("status:", e));
}
setInterval(updateStatus, STATUS_MS);
updateStatus();
</script>
</body>
</html>
""".replace("__HOLD_REPEAT_MS__", str(HOLD_REPEAT_MS)).replace("__STATUS_UPDATE_MS__", str(STATUS_UPDATE_MS))

# ── Flask Routes ──────────────────────────────────────────────────────────────
def ok_json(**kw):     return jsonify({"ok": True,  **kw})
def err_json(m, **kw): return jsonify({"ok": False, "message": m, **kw})

@app.route("/")
def index(): return render_template_string(html)

@app.route("/status")
def status(): return jsonify(get_state_copy())

@app.route("/cmd/<cmd>", methods=["POST"])
def command(cmd):
    cmd = cmd.lower()
    if cmd not in DRIVE_COMMANDS: return err_json("Unknown command")
    cur = get_state_copy()
    if cur["mode_switching"] and cmd != "stop":
        set_state(message="Command ignored: mode switching"); return err_json("Command ignored: mode switching")
    if cur["auto_avoid"]: stop_obstacle_avoidance(send_stop=False, update_message=True)
    cur = get_state_copy()
    if cmd != "stop" and cur["mode"] != "wheel":
        set_state(motion="stop", message="Drive ignored: not wheel mode"); send_to_stm32("stop")
        return err_json("Drive command ignored: not wheel mode")
    send_to_stm32(cmd); set_state(motion=cmd, message=f"Manual: {cmd}")
    return ok_json(message=f"Manual: {cmd}")

@app.route("/mode/<mode>", methods=["POST"])
def mode_route(mode):
    mode = mode.lower()
    if mode not in ("wheel", "leg"): return err_json("Unknown mode")
    ok, msg = start_mode_switch(mode); cur = get_state_copy()
    return jsonify({"ok": ok, "mode": cur["mode"], "mode_switching": cur["mode_switching"], "message": msg})

@app.route("/avoid/start", methods=["POST"])
def avoid_start():
    ok, msg = start_obstacle_avoidance(); return jsonify({"ok": ok, "message": msg})

@app.route("/avoid/stop", methods=["POST"])
def avoid_stop():
    stop_obstacle_avoidance(send_stop=True, update_message=True); return ok_json(message="Obstacle avoidance stopped")

# ── 종료 ──────────────────────────────────────────────────────────────────────
def cleanup():
    try:
        stop_obstacle_avoidance(send_stop=True, update_message=False); time.sleep(0.1); close_lidar()
        if ser and ser.is_open: send_to_stm32("stop"); ser.close(); log_info("[OK] Serial closed")
    except Exception as e: log_info(f"[WARN] Cleanup: {e}")

atexit.register(cleanup)

if __name__ == "__main__":
    init_serial(); ping_stm32(); send_to_stm32("stop"); init_lidar()
    log_info(f"[WEB] http://{WEB_HOST}:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True, use_reloader=False)
