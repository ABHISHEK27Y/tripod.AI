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
    REQUEST_INTERVAL = 6.0      # seconds between calls — respects free tier

    def __init__(self):
        self._client   = None
        self._model    = "llama-3.1-8b-instant"
        self._history: List[Dict] = []
        self._goal     = ""
        self._last     = 0.
        self._loaded   = False

    def load(self):
        try:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY","")
            if not key:
                logger.error("GROQ_API_KEY not set in .env")
                return
            self._client  = Groq(api_key=key)
            self._loaded  = True
            logger.info("Groq LLM agent loaded.")
        except Exception as e:
            logger.error(f"Groq init error: {e}")

    def set_goal(self, goal: str):
        self._goal    = goal.strip()
        self._history = []
        logger.info(f"Agent goal: '{self._goal}'")

    def reason(self, scene: str) -> Optional[Dict[str, Any]]:
        if not self._loaded or not self._client:
            return self._fallback(scene)

        elapsed = time.time() - self._last
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)

        msg = f"Goal: {self._goal}\nScene: {scene}"
        self._history.append({"role":"user","content":msg})
        if len(self._history) > self.MAX_HISTORY*2:
            self._history = self._history[-self.MAX_HISTORY*2:]

        messages = [{"role":"system","content":SYSTEM_PROMPT}] + self._history

        for attempt in range(self.RETRY_LIMIT):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.3,
                    response_format={"type":"json_object"},
                )
                self._last = time.time()
                raw = resp.choices[0].message.content.strip()
                parsed = self._parse(raw)
                if parsed:
                    self._history.append({"role":"assistant","content":raw})
                    return parsed
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1}: {e}")
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
