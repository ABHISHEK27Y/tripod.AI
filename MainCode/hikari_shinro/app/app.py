"""
app.py — Flask + Socket.IO Main Server
=======================================
Hikari Shinro AI · CodeRonin · Ahouba 3.0 · IIIT Manipur
"""

import os
import sys
import threading
import time
import logging
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
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "hikari-shinro-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=False, engineio_logger=False)

detector        = YOLOWorldDetector(model_size="yolov8s-worldv2")
depth_estimator = DepthEstimator()
quant_engine    = QuantEngine()
llm_agent       = HikariAgent()
tts             = TTSEngine(rate=155)
camera          = CameraCapture(device_id=0)

state = {
    "goal":              "",
    "running":           False,
    "detect_next":       ["door", "chair", "person", "obstacle", "steps", "table"],
    "last_action":       "",
    "last_speak":        "",
    "goal_reached":      False,
    "frame_count":       0,
    "agent_interval":    25,
    "connected_clients": 0,
}
state_lock = threading.Lock()


# ── Frame callback ────────────────────────────────────────
def on_frame(jpeg_b64: str, metadata: dict):
    with state_lock:
        state["frame_count"] += 1
        fc = state["frame_count"]
    socketio.emit("frame", {
        "image":       jpeg_b64,
        "detections":  metadata.get("detections", []),
        "scene":       metadata.get("scene", ""),
        "astar_path":  metadata.get("astar_path"),
        "target":      metadata.get("target_label", ""),
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


# ── Vision + agent loop ───────────────────────────────────
def main_loop():
    logger.info("Main vision loop started.")
    frame_index = 0
    while state["running"]:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        with state_lock:
            target         = state["detect_next"][0] if state["detect_next"] else ""
            labels         = list(state["detect_next"])
            agent_interval = state["agent_interval"]
        try:
            annotated, metadata = pipeline.process_frame(frame, target_label=target)
            pipeline.set_labels(labels)
            pipeline.emit_frame(annotated, metadata)
        except Exception as e:
            logger.warning(f"Pipeline error: {e}")
            time.sleep(0.1)
            continue
        frame_index += 1
        if frame_index % agent_interval == 0:
            _call_agent(metadata.get("scene", ""))
        time.sleep(1.0 / 12)
    logger.info("Main vision loop stopped.")


def _call_agent(scene_description: str):
    def _do_call():
        with state_lock:
            if state["goal_reached"]:
                return
        try:
            response = llm_agent.reason(scene_description)
            if response is None:
                return
            with state_lock:
                state["detect_next"]  = response.get("detect_next", state["detect_next"])
                state["last_action"]  = response.get("action", "")
                state["last_speak"]   = response.get("speak", "")
                state["goal_reached"] = response.get("goal_reached", False)
            speak_text = response.get("speak", "")
            if speak_text:
                tts.speak(speak_text, priority=True)
                socketio.emit("agent_speak", {
                    "text":         speak_text,
                    "action":       response.get("action", ""),
                    "goal_reached": response.get("goal_reached", False),
                })
            if response.get("goal_reached"):
                tts.speak(f"Goal reached! You have arrived at {state['goal']}.", priority=True)
                socketio.emit("goal_reached", {"goal": state["goal"]})
        except Exception as e:
            logger.warning(f"Agent call error: {e}")
    threading.Thread(target=_do_call, daemon=True).start()


# ── Voice command callback (called by VoiceController) ────
def on_voice_command(text: str):
    """
    Called every time the user speaks a goal.
    Sets goal, updates state, speaks confirmation, notifies HUD.
    """
    text = text.strip()
    if not text:
        return

    logger.info(f"Voice command received: '{text}'")

    # Set goal in LLM agent
    llm_agent.set_goal(text)

    # Update shared state
    with state_lock:
        state["goal"]         = text
        state["goal_reached"] = False
        goal_word             = text.split()[0].lower()
        state["detect_next"]  = [goal_word, "door", "chair", "person", "obstacle", "steps"]

    # Spoken confirmation — user hears this through speakers
    tts.speak(f"Goal set: {text}. Starting navigation.", priority=True)

    # Update HUD
    socketio.emit("goal_updated", {"goal": text})
    logger.info(f"Goal set via voice: '{text}'")


# ─────────────────────────────────────────────────────────
# HTTP ROUTES
# ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    with state_lock:
        return jsonify({
            "running":       state["running"],
            "goal":          state["goal"],
            "last_action":   state["last_action"],
            "last_speak":    state["last_speak"],
            "goal_reached":  state["goal_reached"],
            "frame_count":   state["frame_count"],
            "detect_labels": state["detect_next"],
        })


@app.route("/api/start", methods=["POST"])
def start_system():
    with state_lock:
        if state["running"]:
            return jsonify({"ok": False, "msg": "Already running"})
        state["running"]     = True
        state["frame_count"] = 0
    threading.Thread(target=main_loop, daemon=True).start()
    return jsonify({"ok": True, "msg": "System started"})


@app.route("/api/stop", methods=["POST"])
def stop_system():
    with state_lock:
        state["running"] = False
    return jsonify({"ok": True, "msg": "System stopped"})


@app.route("/api/goal", methods=["POST"])
def set_goal():
    data = request.get_json(force=True)
    goal = data.get("goal", "").strip()
    if not goal:
        return jsonify({"ok": False, "msg": "Goal text is required"})
    llm_agent.set_goal(goal)
    with state_lock:
        state["goal"]         = goal
        state["goal_reached"] = False
        state["last_speak"]   = f"Understood. Looking for: {goal}"
        goal_word             = goal.split()[0].lower()
        state["detect_next"]  = [goal_word, "door", "chair", "person", "obstacle", "steps"]
    tts.speak(f"Goal set: {goal}. Starting navigation.", priority=True)
    socketio.emit("goal_updated", {"goal": goal})
    return jsonify({"ok": True, "goal": goal})


@app.route("/api/labels", methods=["POST"])
def set_labels():
    data   = request.get_json(force=True)
    labels = data.get("labels", [])
    if not labels:
        return jsonify({"ok": False, "msg": "Labels list required"})
    with state_lock:
        state["detect_next"] = [l.strip().lower() for l in labels[:8] if l.strip()]
    pipeline.set_labels(state["detect_next"])
    return jsonify({"ok": True, "labels": state["detect_next"]})


# ─────────────────────────────────────────────────────────
# SOCKET.IO EVENTS
# ─────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    with state_lock:
        state["connected_clients"] += 1
    emit("connected", {
        "status":  "connected",
        "running": state["running"],
        "goal":    state["goal"],
    })
    logger.info(f"Client connected. Total: {state['connected_clients']}")


@socketio.on("disconnect")
def on_disconnect():
    with state_lock:
        state["connected_clients"] = max(0, state["connected_clients"] - 1)
    logger.info(f"Client disconnected. Total: {state['connected_clients']}")


@socketio.on("voice_goal")
def on_voice_goal(data):
    goal = data.get("goal", "").strip()
    if goal:
        on_voice_command(goal)


@socketio.on("manual_command")
def on_manual_command(data):
    cmd = data.get("command", "").strip().lower()
    if cmd == "stop":
        with state_lock:
            state["running"] = False
        tts.speak("System stopped.")
    elif cmd == "start":
        with state_lock:
            state["running"] = True
        threading.Thread(target=main_loop, daemon=True).start()
        tts.speak("System started.")
    elif cmd.startswith("go to ") or cmd.startswith("find "):
        goal = cmd.replace("go to ", "").replace("find ", "")
        on_voice_command(goal)


# ─────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────
def load_all_models():
    logger.info("=" * 50)
    logger.info("  Hikari Shinro AI — Starting Up")
    logger.info("=" * 50)

    tts.start()
    tts.speak("Loading Hikari Shinro AI systems.")

    llm_agent.load()

    try:
        detector.load()
        tts.speak("Detection system ready.")
    except Exception as e:
        logger.error(f"YOLO-World load failed: {e}")
        tts.speak("Warning: detection system failed to load.")

    try:
        depth_estimator.load()
        tts.speak("Depth system ready.")
    except Exception as e:
        logger.error(f"MiDaS load failed: {e}")
        tts.speak("Warning: depth system failed to load.")

    if not camera.start():
        logger.error("Camera failed to open.")
        tts.speak("Warning: camera not found.")
    else:
        tts.speak("Camera ready.")

    # ── START VOICE CONTROLLER ────────────────────────────
    # This starts the mic listen loop in a background thread.
    # User speaks → Whisper transcribes → on_voice_command() called
    # → goal set → pyttsx3 speaks confirmation → agent navigates
    try:
        voice_controller = VoiceController(
            on_command=on_voice_command,
            tts_rate=155,
            whisper_size="base",
            tts_engine=tts,          # ← pass the global tts
        )
        voice_controller.start()
        logger.info("Voice controller started — mic is listening.")
    except Exception as e:
        logger.error(f"Voice controller failed to start: {e}")
        tts.speak("Warning: voice input unavailable. Use the web interface to set goals.")

    tts.speak("All systems loaded. Hikari Shinro AI is ready. Please speak your navigation goal.")
    logger.info("All models loaded. Starting Flask server.")


if __name__ == "__main__":
    load_all_models()
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )