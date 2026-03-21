import time
import threading
from speech.stt import SpeechToText
from agent.reasoner import AgentReasoner
from speech.tts import speak

class AgentCore:
    def __init__(self, navigator):
        self.stt = SpeechToText()
        self.reasoner = AgentReasoner()
        self.navigator = navigator  # Used to dynamically update target_label
        
        self.current_goal = None
        self.status = "IDLE"  # IDLE, IN_PROGRESS
        
        self.last_reasoning_time = 0
        self.reasoning_interval = 5.0  # evaluate every 5 seconds
        self.world_snapshot = None
        
        self._thread = None
        self._running = False
        
    def start(self):
        self.stt.start_listening()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[AgentCore] Started background reasoning loop.")

    def update_world(self, snapshot):
        self.world_snapshot = snapshot

    def _loop(self):
        while self._running:
            time.sleep(0.5)
            
            # Check for new spoken goals
            new_goal = self.stt.get_latest_goal()
            if new_goal:
                print(f"[Agent] New Goal Received: {new_goal}")
                self.current_goal = new_goal
                self.status = "IN_PROGRESS"
                speak(f"Understood. Starting to look for {new_goal}")
                self.last_reasoning_time = 0  # Force immediate reasoning
                
            # If we have an active goal, reason periodically
            now = time.time()
            if self.status == "IN_PROGRESS" and self.current_goal and self.world_snapshot is not None:
                if now - self.last_reasoning_time > self.reasoning_interval:
                    decision = self.reasoner.decide_next_target(self.current_goal, self.world_snapshot)
                    
                    sub_goal = decision.get("sub_goal")
                    guidance = decision.get("spoken_guidance")
                    status = decision.get("status")
                    
                    # Update FSM Navigator target
                    if sub_goal and sub_goal != self.navigator.target_label:
                        self.navigator.target_label = sub_goal
                        print(f"[Agent] FSM target dynamically updated to: {sub_goal}")
                        
                    if guidance:
                        speak(guidance)
                        
                    if status == "COMPLETED":
                        self.status = "IDLE"
                        self.current_goal = None
                        speak("Goal reached.")
                        print("[Agent] Goal Completed.")
                        
                    self.last_reasoning_time = now

    def stop(self):
        self._running = False
        self.stt.stop_listening()
        if self._thread:
            self._thread.join(timeout=2.0)
