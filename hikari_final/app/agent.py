"""agent.py — LLM Agent Brain · Hikari Shinro AI"""
import json, logging, os, re, time
from typing import Optional, List, Dict, Any
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Hikari — a calm, precise AI navigation assistant for visually impaired users.
You receive a scene description (detected objects, distances, path info) and the user's goal.

Respond with ONLY a valid JSON object — no markdown, no extra text:
{
  "speak": "<max 2 sentences, calm and clear>",
  "detect_next": ["<label1>", "<label2>", "<label3>"],
  "action": "<move_forward|turn_left|turn_right|stop|wait>",
  "goal_reached": false,
  "reasoning": "<brief>"
}

Rules:
- speak in second person: "The door is ahead" not "I see a door"
- mention side: "to your left", "straight ahead", "to your right"
- if obstacle < 0.5m: action must be "stop"
- goal_reached=true only when user is within 0.8m of goal object
- detect_next: include goal object + 2-3 likely obstacles, max 5 labels
- keep speak under 25 words"""


class HikariAgent:
    MAX_HISTORY    = 4
    RETRY_LIMIT    = 2
    REQUEST_INTERVAL = 3.0      # faster for local models

    def __init__(self):
        self._model    = "minimax-m2.5:cloud"  # The model the user just downloaded
        self._history: List[Dict] = []
        self._goal     = ""
        self._last     = 0.
        self._loaded   = False
        self._endpoint = "http://localhost:11434/api/chat"

    def load(self):
        try:
            import requests
            # Pre-flight check to see if Ollama is running
            r = requests.get("http://localhost:11434/")
            if r.status_code == 200:
                self._loaded = True
                logger.info(f"Ollama Agent loaded (model: {self._model})")
            else:
                logger.error("Ollama responded but unexpected status.")
        except Exception as e:
            logger.error(f"Cannot connect to local Ollama. Ensure it is running! Error: {e}")

    def set_goal(self, goal: str):
        self._goal    = goal.strip()
        self._history = []
        logger.info(f"Agent goal: '{self._goal}'")

    def reason(self, scene: str) -> Optional[Dict[str, Any]]:
        if not self._loaded:
            return self._fallback(scene)

        elapsed = time.time() - self._last
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)

        msg = f"Goal: {self._goal}\nScene: {scene}"
        self._history.append({"role":"user","content":msg})
        if len(self._history) > self.MAX_HISTORY*2:
            self._history = self._history[-self.MAX_HISTORY*2:]

        messages = [{"role":"system","content":SYSTEM_PROMPT}] + self._history
        import requests
        
        for attempt in range(self.RETRY_LIMIT):
            try:
                resp = requests.post(self._endpoint, json={
                    "model": self._model,
                    "messages": messages,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.3}
                }, timeout=15)
                self._last = time.time()
                
                if resp.status_code == 200:
                    raw = resp.json()["message"]["content"].strip()
                    parsed = self._parse(raw)
                    if parsed:
                        self._history.append({"role":"assistant","content":raw})
                        return parsed
                else:
                    logger.warning(f"Ollama returned {resp.status_code}: {resp.text}")
                    
            except Exception as e:
                logger.warning(f"Ollama LLM attempt {attempt+1} failed: {e}")
                time.sleep(1.*(attempt+1))

        return self._fallback(scene)

    def _parse(self, raw):
        try:
            clean = re.sub(r"```[a-z]*\n?","",raw).strip()
            d = json.loads(clean)
            if not all(k in d for k in ("speak","detect_next","action","goal_reached")):
                return None
            d["speak"]        = str(d["speak"])[:300]
            d["detect_next"]  = [str(l) for l in d["detect_next"][:6]]
            d["action"]       = str(d["action"])
            d["goal_reached"] = bool(d.get("goal_reached",False))
            return d
        except Exception:
            return None

    def _fallback(self, scene):
        speak="Moving forward carefully."; action="move_forward"
        if "very close" in scene.lower() or "caution" in scene.lower():
            speak="Obstacle very close. Please stop."; action="stop"
        elif self._goal:
            speak=f"Searching for {self._goal}. Moving forward slowly."
        return {
            "speak": speak,
            "detect_next": self._default_labels(),
            "action": action,
            "goal_reached": False,
            "reasoning": "fallback",
        }

    def _default_labels(self):
        base=["door","chair","person","obstacle","steps","table"]
        if self._goal:
            w=self._goal.split()[0].lower()
            if w not in base: base.insert(0,w)
        return base[:6]

    def get_goal(self): return self._goal
