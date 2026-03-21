import eventlet
import eventlet.wsgi
eventlet.monkey_patch() # Must be very top

from flask import Flask, jsonify
from flask_socketio import SocketIO
import logging

app = Flask(__name__)
# Enable CORS and async mode
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Disable default flask logging to avoid clutter
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class HUDState:
    agent_core = None
    navigator = None

@app.route("/")
def index():
    return jsonify({"status": "running", "message": "VisionGuide App Backend"})

@app.route("/status")
def get_status():
    if not HUDState.agent_core:
        return jsonify({"status": "offline"})
    return jsonify({
        "status": "online",
        "current_goal": HUDState.agent_core.current_goal,
        "fsm_target": HUDState.navigator.target_label if HUDState.navigator else None
    })

@socketio.on("connect")
def handle_connect():
    print("[HUD] Frontend Socket Connected.")

def emit_hud_data(world_snapshot, nav_state, current_goal):
    # Sends visual state to Anime HUD over websockets
    socketio.emit("hud_update", {
        "goal": current_goal,
        "target": nav_state,
        "objects": world_snapshot["objects"]
    })

def run_server():
    print("[Server] Starting Flask-SocketIO Server on port 5000")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5000)), app)
