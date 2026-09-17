from flask import Flask, render_template_string, jsonify
import serial, time, threading, atexit, math, os, glob, logging

logging.getLogger("werkzeug").setLevel(logging.ERROR)

try:    import ydlidar; YDLIDAR_OK = True
except: ydlidar = None; YDLIDAR_OK = False

# ── 설정 ──────────────────────────────────────────────────────────────────────
SERIAL_PORT, BAUDRATE = "/dev/stm32", 115200
LIDAR_PORT, LIDAR_BAUDRATE, LIDAR_SAMPLE_RATE = "auto", 230400, 5
LIDAR_SCAN_FREQ, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE = 10.0, 0.28, 16.0
WEB_HOST, WEB_PORT = "0.0.0.0", 5000
HOLD_REPEAT_MS, STATUS_UPDATE_MS, MODE_SWITCH_TIME = 100, 200, 5.0
OBSTACLE_DIST, TOO_CLOSE_DIST = 0.50, 0.12
AVOID_FWD_TIME, AVOID_LOOP_DELAY, AVOID_REPEAT_TIME = 0.08, 0.03, 0.10
FRONT_CENTER_DEG, FRONT_WIDTH_DEG = 180.0, 30.0

CMD_MAP = {
    "ping":"PING", "stop":"STOP", "wheel":"WHEEL", "leg":"LEG",
    "forward":"FWD", "backward":"BACK", "tank_left":"TL", "tank_right":"TR",
}
DRIVE_CMDS = {"forward", "backward", "tank_left", "tank_right", "stop"}

# ── 전역 상태 ─────────────────────────────────────────────────────────────────
app  = Flask(__name__)
ser  = None;  serial_lock = threading.Lock()
laser = None; lidar_lock  = threading.Lock()
avoid_thread = None; avoid_stop = threading.Event()
state_lock = threading.Lock()
state = {"mode":"wheel","motion":"stop","mode_switching":False,
         "auto_avoid":False,"front_distance":None,"message":"Ready"}

def ts():           return time.strftime('%H:%M:%S')
def log(msg):       print(f"[{ts()}] {msg}", flush=True)
def set_state(**kw):
    with state_lock: state.update(kw)
def get_state():
    with state_lock: return dict(state)

# ── Serial ────────────────────────────────────────────────────────────────────
def init_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        time.sleep(2.0); log(f"Serial OK: {SERIAL_PORT}")
        set_state(message=f"Serial opened: {SERIAL_PORT}"); return True
    except Exception as e:
        log(f"Serial ERR: {e}"); set_state(message="Serial open failed"); ser = None; return False

def send(cmd, reply=False):
    cmd = cmd.lower()
    if cmd not in CMD_MAP or ser is None or not ser.is_open: return None
    raw = CMD_MAP[cmd]
    try:
        with serial_lock:
            if reply: ser.reset_input_buffer()
            ser.write((raw + "\r\n").encode()); ser.flush(); log(f"TX→ {raw}")
            if reply:
                r = ser.readline().decode("ascii", errors="ignore").strip()
                log(f"RX← {r}"); return r
    except Exception as e: log(f"Send ERR: {e}")
    return None

def ping():
    ok = send("ping", reply=True) == "OK"
    set_state(message="STM32 OK" if ok else "STM32 PING failed"); return ok

# ── LiDAR ─────────────────────────────────────────────────────────────────────
def find_lidar_ports():
    if LIDAR_PORT != "auto":
        return [LIDAR_PORT] if os.path.exists(LIDAR_PORT) else []
    ports = ["/dev/ydlidar"] if os.path.exists("/dev/ydlidar") else []
    return ports + [p for p in sorted(glob.glob("/dev/ttyUSB*")) if p not in ports]

def init_lidar():
    global laser
    if not YDLIDAR_OK: set_state(message="LiDAR N/A"); return False
    ports = find_lidar_ports()
    if not ports: set_state(message="No LiDAR port"); return False
    try: ydlidar.os_init()
    except: pass
    for port in ports:
        try:
            L = ydlidar.CYdLidar()
            for name, val in [
                ("LidarPropSerialPort",        port),
                ("LidarPropSerialBaudrate",    LIDAR_BAUDRATE),
                ("LidarPropLidarType",         ydlidar.TYPE_TRIANGLE),
                ("LidarPropDeviceType",        ydlidar.YDLIDAR_TYPE_SERIAL),
                ("LidarPropScanFrequency",     LIDAR_SCAN_FREQ),
                ("LidarPropSampleRate",        LIDAR_SAMPLE_RATE),
                ("LidarPropSingleChannel",     False),
                ("LidarPropMaxAngle",          180.0),
                ("LidarPropMinAngle",         -180.0),
                ("LidarPropMaxRange",          LIDAR_MAX_RANGE),
                ("LidarPropMinRange",          LIDAR_MIN_RANGE),
                ("LidarPropIntenstiy",         True),
                ("LidarPropSupportMotorDtrCtrl", False),
            ]:
                if hasattr(ydlidar, name):
                    try: L.setlidaropt(getattr(ydlidar, name), val)
                    except: pass
            # 오타 폴백: Intenstiy 실패 시 Intensity 시도
            if not hasattr(ydlidar, "LidarPropIntenstiy") and hasattr(ydlidar, "LidarPropIntensity"):
                try: L.setlidaropt(ydlidar.LidarPropIntensity, True)
                except: pass
            if not L.initialize(): L.disconnecting(); continue
            if not L.turnOn():    L.turnOff(); L.disconnecting(); continue
            laser = L; log(f"LiDAR OK: {port}"); set_state(message=f"LiDAR: {port}"); return True
        except Exception as e:
            log(f"LiDAR {port}: {e}")
            try: L.disconnecting()
            except: pass
    set_state(message="LiDAR start failed"); return False

def close_lidar():
    global laser
    try:
        if laser: laser.turnOff(); laser.disconnecting(); laser = None
    except Exception as e: log(f"LiDAR close: {e}")

def front_dist():
    if laser is None: return None
    try:
        scan = ydlidar.LaserScan()
        with lidar_lock:
            if not laser.doProcessSimple(scan): return None
        dists = []
        for p in scan.points:
            deg = math.degrees(p.angle) % 360
            d = p.range
            if d <= TOO_CLOSE_DIST or not (LIDAR_MIN_RANGE <= d <= LIDAR_MAX_RANGE): continue
            if abs((deg - FRONT_CENTER_DEG + 180) % 360 - 180) <= FRONT_WIDTH_DEG / 2:
                dists.append(d)
        return min(dists) if dists else None
    except Exception as e: log(f"LiDAR scan: {e}"); return None

# ── 모드 전환 ─────────────────────────────────────────────────────────────────
def _finish_switch(target):
    time.sleep(MODE_SWITCH_TIME)
    set_state(mode=target, mode_switching=False, motion="stop", message=f"Mode: {target}")
    log(f"Mode → {target}")

def start_mode_switch(target):
    target = target.lower()

    if target not in ("wheel", "leg"):
        return False, "Unknown mode"

    cur = get_state()

    if cur["mode_switching"]:
        return False, "Mode switching in progress"

    set_state(
        mode_switching=True,
        auto_avoid=False,
        motion="stop",
        message=f"Switching to {target}"
    )

    stop_avoidance(send_stop=False, update=False)

    # WHEEL / LEG 명령을 무조건 STM32로 전송
    send(target)

    threading.Thread(target=_finish_switch, args=(target,), daemon=True).start()
    return True, f"Switching to {target}"
# ── 장애물 회피 ───────────────────────────────────────────────────────────────
def _avoidance_loop():
    log("Auto avoid started")
    set_state(auto_avoid=True, motion="stop", message="Obstacle avoidance started")
    last_cmd, last_tx = None, 0.0

    def auto_send(cmd, msg):
        nonlocal last_cmd, last_tx
        now = time.time()
        if cmd != last_cmd or (now - last_tx) >= AVOID_REPEAT_TIME:
            send(cmd); last_cmd = cmd; last_tx = now
        set_state(motion=cmd, message=msg)

    while not avoid_stop.is_set():
        cur = get_state()
        if cur["mode"] != "wheel" or cur["mode_switching"]:
            send("stop"); set_state(motion="stop", message="Auto avoid: stopped"); break

        d = front_dist(); set_state(front_distance=d)
        if d is None:
            auto_send("stop", "No LiDAR data"); time.sleep(AVOID_LOOP_DELAY); continue

        if d <= OBSTACLE_DIST:
            log(f"Obstacle: {d:.2f} m → TR")
            auto_send("tank_right", f"Avoiding TR: {d:.2f} m")
        else:
            auto_send("forward", f"Forward: {d:.2f} m")
        time.sleep(AVOID_LOOP_DELAY)

    send("stop"); set_state(auto_avoid=False, motion="stop", message="Obstacle avoidance stopped")
    log("Auto avoid stopped")

def start_avoidance():
    global avoid_thread
    if laser is None:          return False, "LiDAR not ready"
    cur = get_state()
    if cur["mode_switching"]:  return False, "Mode switching in progress"
    if cur["mode"] != "wheel": return False, "Wheel mode only"
    if cur["auto_avoid"]:      return True,  "Already running"
    avoid_stop.clear()
    avoid_thread = threading.Thread(target=_avoidance_loop, daemon=True)
    avoid_thread.start()
    return True, "Obstacle avoidance started"

def stop_avoidance(send_stop=True, update=True):
    global avoid_thread
    avoid_stop.set(); avoid_thread = None
    if send_stop: send("stop")
    if update: set_state(auto_avoid=False, motion="stop", message="Obstacle avoidance stopped")
    else:       set_state(auto_avoid=False, motion="stop")

# ── Web UI ────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><title>Robot Controller</title>
<style>
body{font-family:Arial,sans-serif;text-align:center;margin-top:35px;user-select:none;touch-action:none;background:#f4f4f4}
h1{margin-bottom:20px}
.status{display:inline-block;min-width:360px;padding:15px;margin-bottom:20px;background:white;border-radius:12px;border:1px solid #ccc;font-size:18px;line-height:1.7}
button{width:150px;height:75px;font-size:22px;margin:8px;border-radius:14px;border:1px solid #555;touch-action:none;cursor:pointer}
button:active{background:#ccc} button:disabled{opacity:.45;cursor:not-allowed}
.move{background:#eee} .stop{background:red;color:white;font-weight:bold}
.mode{background:#ddeeff} .auto-on{background:#ddffdd} .auto-off{background:#ffe0cc}
.small{font-size:15px;color:#555;margin-top:15px}
</style></head><body>
<h1>Robot Web Controller</h1>
<div class="status">
  <div>Mode: <b id="mode">-</b></div><div>Motion: <b id="motion">-</b></div>
  <div>Mode Switching: <b id="mode_switching">-</b></div><div>Auto Avoid: <b id="auto_avoid">-</b></div>
  <div>Front Distance: <b id="front_distance">-</b></div><div>Message: <b id="message">-</b></div>
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
<div style="margin-top:20px">
  <button class="auto-on" id="avoid_on" onclick="startAvoid()">Avoid ON</button>
  <button class="auto-off" onclick="stopAvoid()">Avoid OFF</button>
</div>
<p class="small">
  전진/후진/회전은 wheel mode에서만 동작합니다.<br>
  버튼을 누르고 있으면 명령이 반복 전송되고, 떼면 정지합니다.<br>
  장애물 회피는 wheel mode에서만 동작합니다.
</p>
<script>"use strict";
let activeCmd=null,repeatTimer=null,stopping=false;
const REPEAT_MS=__RPT__,STATUS_MS=__STA__;

const post=url=>fetch(url,{method:"POST"}).then(r=>r.json()).then(d=>{upd();return d;});
const sendCmd=(cmd,ref)=>fetch("/cmd/"+cmd,{method:"POST"}).then(r=>r.json()).then(d=>{if(ref!==false)upd();return d;});
const setMode=m=>{stopRpt();sendStop();post("/mode/"+m);};
const startAvoid=()=>{stopRpt();post("/avoid/start");};
const stopAvoid=()=>{stopRpt();post("/avoid/stop");};
const stopAll=()=>{stopRpt();sendStop();};

function stopRpt(){if(repeatTimer){clearInterval(repeatTimer);repeatTimer=null;}activeCmd=null;}
function sendStop(){
  if(stopping)return; stopping=true;
  fetch("/cmd/stop",{method:"POST",keepalive:true}).then(r=>r.json()).then(()=>upd())
  .catch(e=>console.error("stop:",e)).finally(()=>{stopping=false;});
}
function startCmd(cmd,e){
  e.preventDefault();e.stopPropagation();
  try{e.currentTarget.setPointerCapture(e.pointerId);}catch(_){}
  const btn=document.getElementById(cmd);
  if(btn&&btn.disabled)return;
  stopRpt();activeCmd=cmd;stopping=false;
  sendCmd(cmd,true);
  repeatTimer=setInterval(()=>{if(activeCmd)sendCmd(activeCmd,false);},REPEAT_MS);
}
function stopCmd(e){
  e.preventDefault();e.stopPropagation();
  if(!activeCmd)return; stopRpt();sendStop();
}
["forward","backward","tank_left","tank_right"].forEach(id=>{
  const b=document.getElementById(id);
  b.addEventListener("pointerdown",e=>startCmd(id,e));
  b.addEventListener("pointerup",stopCmd);
  b.addEventListener("pointercancel",stopCmd);
  b.addEventListener("lostpointercapture",stopCmd);
  b.addEventListener("pointerleave",e=>{if(activeCmd===id)stopCmd(e);});
  b.addEventListener("contextmenu",e=>e.preventDefault());
});
window.addEventListener("blur",()=>{if(activeCmd){stopRpt();sendStop();}});
document.addEventListener("visibilitychange",()=>{if(document.hidden&&activeCmd){stopRpt();sendStop();}});
document.addEventListener("contextmenu",e=>e.preventDefault());

function upd(){
  fetch("/status").then(r=>r.json()).then(d=>{
    ["mode","motion","mode_switching","auto_avoid","message"].forEach(k=>document.getElementById(k).innerText=d[k]);
    document.getElementById("front_distance").innerText=d.front_distance===null?"None":Number(d.front_distance).toFixed(2)+" m";
    const blocked=d.mode_switching||d.mode!=="wheel"||d.auto_avoid;
    document.querySelectorAll(".drive-btn").forEach(b=>b.disabled=blocked);
    document.getElementById("avoid_on").disabled=d.mode_switching||d.mode!=="wheel";
  }).catch(e=>console.error("status:",e));
}
setInterval(upd,STATUS_MS);upd();
</script></body></html>""".replace("__RPT__", str(HOLD_REPEAT_MS)).replace("__STA__", str(STATUS_UPDATE_MS))

# ── Flask 라우트 ──────────────────────────────────────────────────────────────
ok_r  = lambda **kw: jsonify({"ok": True,  **kw})
err_r = lambda m, **kw: jsonify({"ok": False, "message": m, **kw})

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/status")
def status(): return jsonify(get_state())

@app.route("/cmd/<cmd>", methods=["POST"])
def command(cmd):
    cmd = cmd.lower()
    if cmd not in DRIVE_CMDS: return err_r("Unknown command")
    cur = get_state()
    if cur["mode_switching"] and cmd != "stop":
        set_state(message="Command ignored: mode switching")
        return err_r("Command ignored: mode switching")
    if cur["auto_avoid"]: stop_avoidance(send_stop=False, update=True)
    if cmd != "stop" and get_state()["mode"] != "wheel":
        set_state(motion="stop", message="Drive ignored: not wheel mode"); send("stop")
        return err_r("Drive command ignored: not wheel mode")
    send(cmd); set_state(motion=cmd, message=f"Manual: {cmd}")
    return ok_r(message=f"Manual: {cmd}")

@app.route("/mode/<mode>", methods=["POST"])
def mode_route(mode):
    ok, msg = start_mode_switch(mode.lower()); cur = get_state()
    return jsonify({"ok":ok,"mode":cur["mode"],"mode_switching":cur["mode_switching"],"message":msg})

@app.route("/avoid/start", methods=["POST"])
def avoid_start():
    ok, msg = start_avoidance(); return jsonify({"ok":ok,"message":msg})

@app.route("/avoid/stop", methods=["POST"])
def avoid_stop_route():
    stop_avoidance(send_stop=True, update=True); return ok_r(message="Obstacle avoidance stopped")

# ── 종료 ──────────────────────────────────────────────────────────────────────
def cleanup():
    try:
        stop_avoidance(send_stop=True, update=False); time.sleep(0.1); close_lidar()
        if ser and ser.is_open: send("stop"); ser.close(); log("Serial closed")
    except Exception as e: log(f"Cleanup: {e}")

atexit.register(cleanup)

if __name__ == "__main__":
    init_serial(); ping(); send("stop"); init_lidar()
    log(f"Web: http://{WEB_HOST}:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True, use_reloader=False)
