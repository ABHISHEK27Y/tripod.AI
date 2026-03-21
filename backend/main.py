import eventlet
eventlet.monkey_patch()

import cv2
import time
import numpy as np
import threading

from perception.camera import Camera
from perception.detector import Detector
from perception.depth import DepthEstimator
from perception.spatial import SpatialEstimator
from perception.tracker import IOUTracker
from perception.quant import astar_path, KalmanTracker

from state.world_state import WorldState
from state.navigation_state import NavigationState
from navigation.navigator import NavigatorFSM
from speech.rate_limiter import SpeechLimiter
from speech.tts import speak

from agent.core import AgentCore
from perception.runner import draw

from app import HUDState, emit_hud_data, run_server

def cv_loop():
    print("[Main] Initializing CV thread...")
    
    cam = Camera()
    det = Detector() # YOLO-World
    depth = DepthEstimator() # MiDaS
    spatial = SpatialEstimator()
    tracker = IOUTracker()
    
    # Layer 6 Quant components
    kalman_target = KalmanTracker()

    world = WorldState()
    nav_state = NavigationState()
    navigator = NavigatorFSM(target_label="door") 

    agent = AgentCore(navigator=navigator)
    agent.start()

    # Link to Flask HUD
    HUDState.agent_core = agent
    HUDState.navigator = navigator

    speech_limiter = SpeechLimiter()

    cam.open()
    prev = time.time()
    
    print("[Main] System active. Speak a goal to begin navigation.")
    speak("System active. What is your destination?")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            # If Agent found a new sub-goal, tell YOLO-World to find it
            current_target = navigator.target_label
            if current_target and current_target not in det.current_classes:
                det.set_target_classes([current_target, "person", "chair", "door"])

            # 1. Perception
            detections = det.detect(frame)
            tracks = tracker.update(detections)

            tracked_dets = [
                {"id": t.id, "bbox": t.bbox, "label": t.label, "confidence": 1.0}
                for t in tracks
            ]

            depth_map = depth.estimate(frame)
            spatial_objects = spatial.estimate(tracked_dets, depth_map)

            # 2. Layer 6 Quant: A* Pathfinding (Sample vertical safe path computation)
            # Create a heavily downsampled 32x32 occupancy grid for speed
            small_depth = cv2.resize(depth_map, (32, 32))
            grid = (small_depth > 0.6).astype(int) # 1 if object is too close
            # Start at bottom center, goal at top center
            safe_path = astar_path(grid, (31, 16), (0, 16))

            # 3. World State
            world.update(spatial_objects, target_label=current_target)
            world_snapshot = world.snapshot()
            agent.update_world(world_snapshot)

            # 4. Navigation FSM
            decision = navigator.decide(world_snapshot, nav_state)
            nav_snapshot = nav_state.snapshot()

            if decision:
                action = decision['action']
                if speech_limiter.should_speak(action):
                    speak(f"{action}. {decision['reason']}")

            # 5. Layer 8: Push to Anime HUD
            emit_hud_data(world_snapshot, nav_snapshot, agent.current_goal)

            # 6. Debug Visualization
            draw(frame, world_snapshot, nav_snapshot, decision)
            now = time.time()
            fps = 1 / (now - prev)
            prev = now

            cv2.putText(frame, f"FPS {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"GOAL: {agent.current_goal}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"TARGET: {current_target}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Agentic Navigation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                speak("Shutting down.")
                break

    finally:
        agent.stop()
        cam.release()
        import os
        os._exit(0) # Force kill Flask server if cv2 loop exits


def main():
    # Start vision/agent processing in background thread
    t = threading.Thread(target=cv_loop, daemon=True)
    t.start()
    
    # Run Flask-SocketIO Eventlet server on main thread
    run_server()

if __name__ == "__main__":
    main()
