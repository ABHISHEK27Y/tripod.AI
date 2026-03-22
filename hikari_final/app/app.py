"""
app.py — Hikari Shinro AI Main Server
CodeRonin · Ahouba 3.0 · IIIT Manipur · PS-01

Run:  python app.py
Open: http://localhost:5000
Speak: "Find the chair" — system listens, detects, guides via speaker
"""

import os, sys, threading, time, logging
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hikari.app")

sys.path.insert(0, os.path.dirname(__file__))
from detection import YOLOWorldDetector
from depth     import DepthEstimator
from quant     import QuantEngine
from agent     import HikariAgent
from speech    import TTSEngine, VoiceController
from vision    import CameraCapture, FramePipeline

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder  =os.path.join(os.path.dirname(__file__), "..", "static"),
)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "hikari-2026")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=False, engineio_logger=False)

# ── Components ────────────────────────────────────────────
detector        = YOLOWorldDetector()
depth_estimator = DepthEstimator()
quant_engine    = QuantEngine()
llm_agent       = HikariAgent()
tts             = TTSEngine(rate=155)      # SINGLE TTS instance shared everywhere
camera          = CameraCapture(device_id=int(os.getenv("CAMERA_DEVICE","0")))

state = {
    "goal":            "",
    "running":         False,
    "detect_next":     ["door","chair","person","obstacle","steps","table"],
    "last_action":     "",
    "last_speak":      "",
    "goal_reached":    False,
    "frame_count":     0,
    "agent_interval":  25,          # call LLM every 25 frames ≈ every 2 secs
    "connected":       0,
    "last_fast_speak": "",
    "last_fast_time":  0.0,
}
lock = threading.Lock()


# ── Frame emit ────────────────────────────────────────────
def on_frame(jpeg_b64, meta):
    with lock:
        state["frame_count"] += 1
        fc = state["frame_count"]
    socketio.emit("frame", {
        "image":       jpeg_b64,
        "detections":  meta.get("detections",[]),
        "scene":       meta.get("scene",""),
        "astar_path":  meta.get("astar_path"),
        "target":      meta.get("target_label",""),
        "goal":        state["goal"],
        "action":      state["last_action"],
        "speak":       state["last_speak"],
        "frame_count": fc,
    })


pipeline = FramePipeline(
    detector=detector,
    depth_estimator=depth_estimator,
    quant_engine=quant_engine,
    on_frame=on_frame,
)


# ── Vision loop ───────────────────────────────────────────
def main_loop():
    logger.info("Vision loop started.")
    fi = 0
    while state["running"]:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05); continue
        with lock:
            target = state["detect_next"][0] if state["detect_next"] else ""
            labels = list(state["detect_next"])
            interval = state["agent_interval"]
        try:
            pipeline.set_labels(labels)
            annotated, meta = pipeline.process_frame(frame, target_label=target)
            pipeline.emit_frame(annotated, meta)
        except Exception as e:
            logger.warning(f"Pipeline: {e}"); time.sleep(0.1); continue
            
        scene_str = meta.get("scene", "")
        if "Navigation: " in scene_str:
            path_instr = scene_str.split("Navigation: ")[-1].strip(" .")
            if path_instr:
                now = time.time()
                with lock:
                    if (path_instr != state["last_fast_speak"] and (now - state["last_fast_time"] > 1.5)) or (now - state["last_fast_time"] > 3.0):
                        if path_instr != "path clear, proceed":
                            tts.speak(path_instr, priority=False)
                            state["last_fast_speak"] = path_instr
                            state["last_fast_time"]  = now

        fi += 1
        if fi % interval == 0:
            _agent_call(meta.get("scene",""))
        time.sleep(1./12)
    logger.info("Vision loop stopped.")


def _agent_call(scene):
    def _run():
        with lock:
            if state["goal_reached"]: return
        try:
            r = llm_agent.reason(scene)
            if not r: return
            with lock:
                state["detect_next"]  = r.get("detect_next", state["detect_next"])
                state["last_action"]  = r.get("action","")
                state["last_speak"]   = r.get("speak","")
                state["goal_reached"] = r.get("goal_reached", False)
            speak = r.get("speak","")
            if speak:
                tts.speak(speak, priority=True)          # speak through speakers
                socketio.emit("agent_speak",{
                    "text": speak,
                    "action": r.get("action",""),
                    "goal_reached": r.get("goal_reached",False),
                })
            if r.get("goal_reached"):
                msg = f"Goal reached! You have arrived at {state['goal']}."
                tts.speak(msg, priority=True)
                socketio.emit("goal_reached",{"goal": state["goal"]})
        except Exception as e:
            logger.warning(f"Agent: {e}")
    threading.Thread(target=_run, daemon=True).start()


# ── Voice command handler ─────────────────────────────────
def on_voice(text: str):
    """Called when user speaks. Sets goal, confirms via speaker, starts navigation."""
    text = text.strip()
    if not text: return
    logger.info(f"Voice command: '{text}'")
    llm_agent.set_goal(text)
    with lock:
        state["goal"]         = text
        state["goal_reached"] = False
        w = text.split()[0].lower()
        state["detect_next"]  = [w,"door","chair","person","obstacle","steps"]
    # Confirm via speaker so blind user knows it worked
    tts.speak(f"Goal set: {text}. Starting navigation.", priority=True)
    socketio.emit("goal_updated", {"goal": text})


# ── Routes ────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/status")
def status():
    with lock: return jsonify({
        "running": state["running"], "goal": state["goal"],
        "last_speak": state["last_speak"], "goal_reached": state["goal_reached"],
        "frame_count": state["frame_count"],
    })

@app.route("/api/start", methods=["POST"])
def start():
    with lock:
        if state["running"]: return jsonify({"ok":False,"msg":"Already running"})
        state["running"]=True; state["frame_count"]=0
    threading.Thread(target=main_loop, daemon=True).start()
    tts.speak("Navigation started.", priority=True)
    return jsonify({"ok":True})

@app.route("/api/stop", methods=["POST"])
def stop():
    with lock: state["running"]=False
    tts.speak("Navigation stopped.", priority=True)
    return jsonify({"ok":True})

@app.route("/api/goal", methods=["POST"])
def set_goal():
    data = request.get_json(force=True)
    goal = data.get("goal","").strip()
    if not goal: return jsonify({"ok":False})
    on_voice(goal)
    return jsonify({"ok":True, "goal": goal})


# ── Socket.IO ─────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    with lock: state["connected"]+=1
    emit("connected",{"running":state["running"],"goal":state["goal"]})

@socketio.on("disconnect")
def on_disconnect():
    with lock: state["connected"]=max(0,state["connected"]-1)

@socketio.on("voice_goal")
def on_voice_goal(data):
    goal=data.get("goal","").strip()
    if goal: on_voice(goal)


# ── Startup ───────────────────────────────────────────────
def startup():
    logger.info("="*50)
    logger.info("  Hikari Shinro AI — Starting Up")
    logger.info("="*50)

    tts.start()
    tts.speak("Loading Hikari Shinro AI.")

    llm_agent.load()

    try:
        detector.load()
        tts.speak("Detection ready.")
    except Exception as e:
        logger.error(f"YOLO failed: {e}")
        tts.speak("Warning: detection unavailable.")

    try:
        depth_estimator.load()
        tts.speak("Depth system ready.")
    except Exception as e:
        logger.error(f"MiDaS failed: {e}")
        tts.speak("Warning: depth unavailable.")

    if not camera.start():
        logger.error("Camera failed.")
        tts.speak("Warning: camera not found.")
    else:
        tts.speak("Camera ready.")

    # Start voice controller — shares the SAME tts instance (no conflict)
    try:
        vc = VoiceController(
            on_command=on_voice,
            whisper_size="base",
            tts_engine=tts,             # pass existing tts — prevents double-start error
        )
        vc.start()
        logger.info("Voice controller started.")
    except Exception as e:
        logger.error(f"Voice failed: {e}")
        tts.speak("Warning: voice input unavailable. Use the web interface.")

    tts.speak("All systems ready. Please speak your navigation goal.")
    logger.info("Startup complete.")


if __name__ == "__main__":
    startup()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
