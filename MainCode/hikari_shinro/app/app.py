"""
app.py — Flask + Socket.IO Main Server
=======================================
Hikari Shinro AI · CodeRonin · Ahouba 3.0 · IIIT Manipur

Coordinates all layers:
  Camera → Detection + Depth → Quant Engine → LLM Agent → TTS + HUD

Run:  python app.py
Open: http://localhost:5000
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

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hikari.app")

# ── Add app/ to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from detection import YOLOWorldDetector
from depth     import DepthEstimator
from quant     import QuantEngine
from agent     import HikariAgent
from speech    import TTSEngine
from vision    import CameraCapture, FramePipeline

# ── Flask & Socket.IO ─────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder  =os.path.join(os.path.dirname(__file__), "..", "static"),
)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "hikari-shinro-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", logger=False, engineio_logger=False)

# ── Component instances ───────────────────────────────────
detector = YOLOWorldDetector(model_size="yolov8s-worldv2")
depth_estimator = DepthEstimator()
quant_engine = QuantEngine()
llm_agent = HikariAgent()
tts = TTSEngine(rate=155)
camera = CameraCapture(device_id=0)

# ── Shared state ──────────────────────────────────────────
state = {
    "goal":            "",
    "running":         False,
    "detect_next":     ["door", "chair", "person", "obstacle", "steps", "table"],
    "last_action":     "",
    "last_speak":      "",
    "goal_reached":    False,
    "frame_count":     0,
    "agent_interval":  15,    # call LLM every N frames
    "connected_clients": 0,
}
state_lock = threading.Lock()


# ── Frame callback (called from pipeline per frame) ───────
def on_frame(jpeg_b64: str, metadata: dict):
    """Called every frame — emits to all connected HUD clients."""
    with state_lock:
        state["frame_count"] += 1
        fc = state["frame_count"]

    socketio.emit("frame", {
        "image":      jpeg_b64,
        "detections": metadata.get("detections", []),
        "scene":      metadata.get("scene", ""),
        "astar_path": metadata.get("astar_path"),
        "target":     metadata.get("target_label", ""),
        "goal":       state["goal"],
        "action":     state["last_action"],
        "speak":      state["last_speak"],
        "frame_count": fc,
    })


# ── Frame pipeline ────────────────────────────────────────
pipeline = FramePipeline(
    detector=detector,
    depth_estimator=depth_estimator,
    quant_engine=quant_engine,
    on_frame=on_frame,
)


# ── Main vision + agent loop ──────────────────────────────
def main_loop():
    """
    Core agentic loop running in background thread.
    Every frame: detect + depth + quant → every N frames: call LLM agent.
    """
    logger.info("Main vision loop started.")
    frame_index = 0

    while state["running"]:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        with state_lock:
            target = state["detect_next"][0] if state["detect_next"] else ""
            labels = list(state["detect_next"])
            agent_interval = state["agent_interval"]

        # Run pipeline
        try:
            annotated, metadata = pipeline.process_frame(frame, target_label=target)
            pipeline.set_labels(labels)
            pipeline.emit_frame(annotated, metadata)
        except Exception as e:
            logger.warning(f"Pipeline error: {e}")
            time.sleep(0.1)
            continue

        # Call LLM agent every N frames
        frame_index += 1
        if frame_index % agent_interval == 0:
            _call_agent(metadata.get("scene", ""))

        # Maintain target FPS
        time.sleep(1.0 / 12)

    logger.info("Main vision loop stopped.")


def _call_agent(scene_description: str):
    """Call LLM agent in a separate thread to avoid blocking the frame loop."""
    def _do_call():
        with state_lock:
            if state["goal_reached"]:
                return
        try:
            response = llm_agent.reason(scene_description)
            if response is None:
                return

            with state_lock:
                state["detect_next"] = response.get("detect_next", state["detect_next"])
                state["last_action"] = response.get("action", "")
                state["last_speak"]  = response.get("speak", "")
                state["goal_reached"] = response.get("goal_reached", False)

            # Speak guidance
            speak_text = response.get("speak", "")
            if speak_text:
                tts.speak(speak_text)
                socketio.emit("agent_speak", {
                    "text":   speak_text,
                    "action": response.get("action", ""),
                    "goal_reached": response.get("goal_reached", False),
                })

            if response.get("goal_reached"):
                tts.speak(f"Goal reached! You have arrived at {state['goal']}.", priority=True)
                socketio.emit("goal_reached", {"goal": state["goal"]})
                logger.info(f"Goal reached: {state['goal']}")

        except Exception as e:
            logger.warning(f"Agent call error: {e}")

    threading.Thread(target=_do_call, daemon=True).start()


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
            "running":      state["running"],
            "goal":         state["goal"],
            "last_action":  state["last_action"],
            "last_speak":   state["last_speak"],
            "goal_reached": state["goal_reached"],
            "frame_count":  state["frame_count"],
            "detect_labels": state["detect_next"],
        })


@app.route("/api/start", methods=["POST"])
def start_system():
    """Start the vision loop."""
    with state_lock:
        if state["running"]:
            return jsonify({"ok": False, "msg": "Already running"})
        state["running"] = True
        state["frame_count"] = 0

    t = threading.Thread(target=main_loop, daemon=True)
    t.start()
    return jsonify({"ok": True, "msg": "System started"})


@app.route("/api/stop", methods=["POST"])
def stop_system():
    with state_lock:
        state["running"] = False
    return jsonify({"ok": True, "msg": "System stopped"})


@app.route("/api/goal", methods=["POST"])
def set_goal():
    """Set navigation goal from web UI text input or voice."""
    data = request.get_json(force=True)
    goal = data.get("goal", "").strip()
    if not goal:
        return jsonify({"ok": False, "msg": "Goal text is required"})

    llm_agent.set_goal(goal)
    with state_lock:
        state["goal"] = goal
        state["goal_reached"] = False
        state["last_speak"] = f"Understood. Looking for: {goal}"
        # Prime detect_next with likely labels from goal
        goal_word = goal.split()[0].lower()
        state["detect_next"] = [goal_word, "door", "chair", "person", "obstacle", "steps"]

    tts.speak(f"Goal set: {goal}. Starting navigation.")
    socketio.emit("goal_updated", {"goal": goal})
    return jsonify({"ok": True, "goal": goal})


@app.route("/api/labels", methods=["POST"])
def set_labels():
    """Manually override detection labels."""
    data = request.get_json(force=True)
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
        "status": "connected",
        "running": state["running"],
        "goal": state["goal"],
    })
    logger.info(f"Client connected. Total: {state['connected_clients']}")


@socketio.on("disconnect")
def on_disconnect():
    with state_lock:
        state["connected_clients"] = max(0, state["connected_clients"] - 1)
    logger.info(f"Client disconnected. Total: {state['connected_clients']}")


@socketio.on("voice_goal")
def on_voice_goal(data):
    """Receive transcribed goal text from frontend voice button."""
    goal = data.get("goal", "").strip()
    if goal:
        llm_agent.set_goal(goal)
        with state_lock:
            state["goal"] = goal
            state["goal_reached"] = False
        tts.speak(f"Goal set: {goal}")
        emit("goal_updated", {"goal": goal}, broadcast=True)


@socketio.on("manual_command")
def on_manual_command(data):
    """Accept typed commands from the HUD input box."""
    cmd = data.get("command", "").strip().lower()
    if cmd == "stop":
        with state_lock:
            state["running"] = False
        tts.speak("System stopped.")
    elif cmd == "start":
        with state_lock:
            state["running"] = True
        t = threading.Thread(target=main_loop, daemon=True)
        t.start()
        tts.speak("System started.")
    elif cmd.startswith("go to ") or cmd.startswith("find "):
        goal = cmd.replace("go to ", "").replace("find ", "")
        llm_agent.set_goal(goal)
        with state_lock:
            state["goal"] = goal
            state["goal_reached"] = False
        tts.speak(f"Searching for {goal}.")


# ─────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────
def load_all_models():
    """Load all ML models in sequence (startup phase)."""
    logger.info("=" * 50)
    logger.info("  Hikari Shinro AI — Starting Up")
    logger.info("=" * 50)

    # TTS first — gives spoken feedback during loading
    tts.start()
    tts.speak("Loading Hikari Shinro AI systems.")

    # LLM agent
    llm_agent.load()

    # Object detection
    try:
        detector.load()
        tts.speak("Detection system ready.")
    except Exception as e:
        logger.error(f"YOLO-World load failed: {e}")
        tts.speak("Warning: detection system failed to load.")

    # Depth estimation
    try:
        depth_estimator.load()
        tts.speak("Depth system ready.")
    except Exception as e:
        logger.error(f"MiDaS load failed: {e}")
        tts.speak("Warning: depth system failed to load.")

    # Camera
    if not camera.start():
        logger.error("Camera failed to open. Check device.")
        tts.speak("Warning: camera not found.")
    else:
        tts.speak("Camera ready.")

    tts.speak("All systems loaded. Hikari Shinro AI is ready. Please set a navigation goal.")
    logger.info("All models loaded. Starting Flask server.")


if __name__ == "__main__":
    load_all_models()
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,  # required on Windows
    )
