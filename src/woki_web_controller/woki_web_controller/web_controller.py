import logging
import threading

from flask import Flask, jsonify, render_template_string
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
HOLD_REPEAT_MS = 100
STATUS_UPDATE_MS = 200

MODE_COMMANDS = {"wheel", "leg"}
MANUAL_COMMANDS = {"forward", "backward", "left", "right", "stop"}


HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><title>WoKi Robot Controller</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body{font-family:Arial,sans-serif;text-align:center;margin-top:35px;user-select:none;touch-action:pan-y;overflow-y:auto;background:#f4f4f4}
h1{margin-bottom:20px}
.status{display:inline-block;min-width:360px;padding:15px;margin-bottom:20px;background:white;border-radius:12px;border:1px solid #ccc;font-size:18px;line-height:1.7}
button{width:150px;height:75px;font-size:22px;margin:8px;border-radius:14px;border:1px solid #555;touch-action:none;cursor:pointer}
button:active{background:#ccc} button:disabled{opacity:.45;cursor:not-allowed}
.move{background:#eee} .stop{background:red;color:white;font-weight:bold}
.mode{background:#ddeeff} .mode.selected{background:#78aef5;color:white;font-weight:bold}
.feature{background:#ddffdd} .feature.selected{background:#57c96f;color:white;font-weight:bold}
.feature-row{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:16px;margin-top:20px}
.feature-row .feature{flex:0 0 150px;width:150px;height:75px;margin:0;white-space:nowrap}
.small{font-size:15px;color:#555;margin-top:15px}
@media (max-width:600px){
body{margin:16px 8px 24px;min-height:100vh}
h1{font-size:24px;margin:0 0 14px}
.status{display:block;min-width:0;max-width:420px;box-sizing:border-box;margin:0 auto 12px;padding:10px;font-size:16px}
button{width:calc((100vw - 48px)/3);max-width:150px;height:68px;margin:4px;font-size:16px}
.feature-row{box-sizing:border-box;gap:8px;margin-top:12px;padding:0 4px}
.feature-row .feature{flex:0 1 calc((100% - 16px)/3);width:auto;min-width:88px;max-width:150px;height:68px;margin:0;padding:0 4px;font-size:15px;box-sizing:border-box}
.small{margin:12px 8px 24px;font-size:13px}
}
</style></head><body>
<h1>WoKi Robot Web Controller</h1>
<div class="status">
  <div>Mode: <b id="mode">UNKNOWN</b></div>
  <div>State: <b id="state">WAIT_CONTROL_MANAGER</b></div>
  <div>Ready: <b id="ready">false</b></div>
  <div>Active Feature: <b id="active_feature">none</b></div>
  <div>Ramp Drive: <b id="ramp_drive_active">OFF</b></div>
</div>
<div>
  <button class="mode control-btn" id="mode-wheel" onclick="setMode('wheel')" disabled>WHEEL</button>
  <button class="mode control-btn" id="mode-leg" onclick="setMode('leg')" disabled>LEG</button>
</div>
<div><button class="move drive-btn control-btn" id="forward" disabled>Forward</button></div>
<div>
  <button class="move drive-btn control-btn" id="left" disabled>Left</button>
  <button class="stop" onclick="stopAll()">STOP</button>
  <button class="move drive-btn control-btn" id="right" disabled>Right</button>
</div>
<div><button class="move drive-btn control-btn" id="backward" disabled>Backward</button></div>
<div class="feature-row">
  <button class="feature control-btn" id="ramp-drive" onclick="toggleRamp()" disabled>경사로 주행</button>
  <button class="feature control-btn" onclick="startAvoid()" disabled>장애물 회피</button>
  <button class="feature control-btn" onclick="startStep()" disabled>단차 극복</button>
</div>
<p class="small">
  방향 버튼을 누르고 있으면 사용자 intent가 반복 전송되고, 떼면 정지합니다.<br>
  실제 mode와 동작 상태는 Control Manager 상태를 표시합니다.
</p>
<script>"use strict";
let activeCmd=null,repeatTimer=null,stopping=false;
const REPEAT_MS=__RPT__,STATUS_MS=__STA__;

const post=url=>fetch(url,{method:"POST"}).then(r=>r.json()).then(d=>{upd();return d;});
const sendCmd=(cmd,refresh)=>fetch("/cmd/"+cmd,{method:"POST"}).then(r=>r.json()).then(d=>{if(refresh!==false)upd();return d;});
const setMode=mode=>{stopRpt();post("/mode/"+mode);};
const startAvoid=()=>{stopRpt();post("/avoid/start");};
const startStep=()=>{stopRpt();post("/step/start");};
const toggleRamp=()=>{stopRpt();post("/ramp/toggle");};
const stopAll=()=>{stopRpt();sendStop();};

function stopRpt(){if(repeatTimer){clearInterval(repeatTimer);repeatTimer=null;}activeCmd=null;}
function sendStop(){
  if(stopping)return;stopping=true;
  fetch("/cmd/stop",{method:"POST",keepalive:true}).then(r=>r.json()).then(()=>upd())
  .catch(e=>console.error("stop:",e)).finally(()=>{stopping=false;});
}
function startCmd(cmd,event){
  event.preventDefault();event.stopPropagation();
  try{event.currentTarget.setPointerCapture(event.pointerId);}catch(_){}
  const button=document.getElementById(cmd);
  if(button&&button.disabled)return;
  stopRpt();activeCmd=cmd;stopping=false;
  sendCmd(cmd,true);
  repeatTimer=setInterval(()=>{if(activeCmd)sendCmd(activeCmd,false);},REPEAT_MS);
}
function stopCmd(event){
  event.preventDefault();event.stopPropagation();
  if(!activeCmd)return;stopRpt();sendStop();
}
["forward","backward","left","right"].forEach(id=>{
  const button=document.getElementById(id);
  button.addEventListener("pointerdown",event=>startCmd(id,event));
  button.addEventListener("pointerup",stopCmd);
  button.addEventListener("pointercancel",stopCmd);
  button.addEventListener("lostpointercapture",stopCmd);
  button.addEventListener("pointerleave",event=>{if(activeCmd===id)stopCmd(event);});
  button.addEventListener("contextmenu",event=>event.preventDefault());
});
window.addEventListener("blur",()=>{if(activeCmd){stopRpt();sendStop();}});
document.addEventListener("visibilitychange",()=>{if(document.hidden&&activeCmd){stopRpt();sendStop();}});
document.addEventListener("contextmenu",event=>event.preventDefault());

function upd(){
  fetch("/status").then(r=>r.json()).then(data=>{
    document.getElementById("mode").innerText=data.mode;
    document.getElementById("state").innerText=data.state;
    document.getElementById("ready").innerText=String(data.ready);
    document.getElementById("active_feature").innerText=data.active_feature;
    const mode=String(data.mode).toLowerCase();
    const rampActive=Boolean(data.ramp_drive_active);
    const rampButton=document.getElementById("ramp-drive");
    document.getElementById("ramp_drive_active").innerText=rampActive?"ON":"OFF";
    rampButton.classList.toggle("selected",rampActive);
    document.getElementById("mode-wheel").classList.toggle("selected",mode==="wheel");
    document.getElementById("mode-leg").classList.toggle("selected",mode==="leg");
    document.querySelectorAll(".control-btn").forEach(button=>button.disabled=!data.ready);
    document.getElementById("left").disabled=!data.ready||rampActive;
    document.getElementById("right").disabled=!data.ready||rampActive;
    rampButton.disabled=!data.ready||mode!=="wheel"||data.active_feature!=="none";
  }).catch(error=>console.error("status:",error));
}
setInterval(upd,STATUS_MS);upd();
</script></body></html>""".replace("__RPT__", str(HOLD_REPEAT_MS)).replace(
    "__STA__", str(STATUS_UPDATE_MS)
)


class WokiWebController(Node):
    def __init__(self):
        super().__init__("woki_web_controller")

        self.mode_cmd_pub = self.create_publisher(
            String,
            "/woki/web/mode_cmd",
            10,
        )
        self.manual_cmd_pub = self.create_publisher(
            String,
            "/woki/web/manual_cmd",
            10,
        )
        self.feature_cmd_pub = self.create_publisher(
            String,
            "/woki/web/feature_cmd",
            10,
        )

        self.mode_sub = self.create_subscription(
            String,
            "/woki/control/mode",
            self.mode_callback,
            10,
        )
        self.state_sub = self.create_subscription(
            String,
            "/woki/control/state",
            self.state_callback,
            10,
        )
        self.ready_sub = self.create_subscription(
            Bool,
            "/woki/control/ready",
            self.ready_callback,
            10,
        )
        self.active_feature_sub = self.create_subscription(
            String,
            "/woki/control/active_feature",
            self.active_feature_callback,
            10,
        )
        self.ramp_drive_active_sub = self.create_subscription(
            Bool,
            "/woki/control/ramp_drive_active",
            self.ramp_drive_active_callback,
            10,
        )

        self.status_lock = threading.Lock()
        self.status = {
            "mode": "UNKNOWN",
            "state": "WAIT_CONTROL_MANAGER",
            "ready": False,
            "active_feature": "none",
            "ramp_drive_active": False,
        }

    def mode_callback(self, msg):
        self.update_status(mode=msg.data)

    def state_callback(self, msg):
        self.update_status(state=msg.data)

    def ready_callback(self, msg):
        self.update_status(ready=bool(msg.data))

    def active_feature_callback(self, msg):
        self.update_status(active_feature=msg.data)

    def ramp_drive_active_callback(self, msg):
        self.update_status(ramp_drive_active=bool(msg.data))

    def update_status(self, **values):
        with self.status_lock:
            self.status.update(values)

    def get_status(self):
        with self.status_lock:
            return dict(self.status)

    def publish_mode_cmd(self, mode):
        self.mode_cmd_pub.publish(String(data=mode))

    def publish_manual_cmd(self, cmd):
        self.manual_cmd_pub.publish(String(data=cmd))

    def publish_feature_cmd(self, feature):
        self.feature_cmd_pub.publish(String(data=feature))


def create_flask_app(node):
    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    def ok_response(**values):
        return jsonify({"ok": True, **values})

    def error_response(message):
        return jsonify({"ok": False, "message": message}), 400

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/status")
    def status():
        return jsonify(node.get_status())

    @app.route("/cmd/<cmd>", methods=["POST"])
    def command(cmd):
        cmd = cmd.lower()
        if cmd not in MANUAL_COMMANDS:
            return error_response("Unknown manual command")

        node.publish_manual_cmd(cmd)
        return ok_response(command=cmd)

    @app.route("/mode/<mode>", methods=["POST"])
    def mode_command(mode):
        mode = mode.lower()
        if mode not in MODE_COMMANDS:
            return error_response("Unknown mode command")

        node.publish_mode_cmd(mode)
        return ok_response(mode_request=mode)

    @app.route("/avoid/start", methods=["POST"])
    def start_avoidance():
        node.publish_feature_cmd("obstacle_avoid")
        return ok_response(feature="obstacle_avoid")

    @app.route("/step/start", methods=["POST"])
    def start_step_overcome():
        node.publish_feature_cmd("step_overcome")
        return ok_response(feature="step_overcome")

    @app.route("/ramp/toggle", methods=["POST"])
    def toggle_ramp_drive():
        node.publish_feature_cmd("ramp_drive")
        return ok_response(feature="ramp_drive")

    return app


def run_flask(app):
    app.run(
        host=WEB_HOST,
        port=WEB_PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


def main(args=None):
    rclpy.init(args=args)
    node = WokiWebController()
    app = create_flask_app(node)
    flask_thread = threading.Thread(
        target=run_flask,
        args=(app,),
        daemon=True,
        name="woki_web_flask",
    )
    flask_thread.start()

    node.get_logger().info(
        f"WoKi Web Controller started | http://{WEB_HOST}:{WEB_PORT}"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
