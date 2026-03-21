"""
agent.py — LLM Agent Brain (Groq · LLaMA 3)
=============================================
Hikari Shinro AI · CodeRonin · Ahouba 3.0 · IIIT Manipur

The agentic core: receives scene context → decomposes goals →
generates navigation guidance → loops without re-prompting.

Agent response JSON schema:
{
  "speak":        "Spoken guidance for the user",
  "detect_next":  ["label1", "label2", ...],
  "action":       "move_forward | turn_left | turn_right | stop | wait",
  "goal_reached": false,
  "reasoning":    "Internal chain-of-thought (not spoken)"
}
"""

import json
import logging
import os
import re
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Hikari — a calm, precise AI navigation assistant
for visually impaired users. You receive a scene description (detected objects,
their distances, and A* path info) and the user's current goal. You must:

1. Give a CONCISE spoken navigation instruction (max 2 sentences).
   Prioritise safety: obstacles < 0.8m always get mentioned first.
2. Return an updated list of objects to detect next frame.
3. Decide the navigation action.
4. Say goal_reached=true only when the user is within 0.6m of their goal object.

Always respond with ONLY a valid JSON object — no markdown, no extra text:
{
  "speak":        "<spoken guidance — calm, clear, max 2 sentences>",
  "detect_next":  ["<label1>", "<label2>", ...],
  "action":       "<move_forward|turn_left|turn_right|stop|wait>",
  "goal_reached": false,
  "reasoning":    "<brief chain-of-thought>"
}

Rules:
- NEVER use markdown in speak field
- detect_next should include goal object + likely obstacles (3-6 labels total)
- If obstacle < 0.5m: action must be "stop" or "turn_left"/"turn_right"
- Speak in second person: "The door is ahead" not "I see a door"
- Be specific about side: "to your left", "straight ahead", "to your right"
"""


class HikariAgent:
    """
    LLM-powered agentic navigation brain.
    Maintains conversation history for multi-turn reasoning.
    """

    MAX_HISTORY = 6          # keep last N turns to avoid context overflow
    RETRY_LIMIT = 2
    REQUEST_INTERVAL = 6.0   # min seconds between LLM calls

    def __init__(self):
        self._client       = None
        self._model        = "llama-3.1-8b-instant"
        self._history: List[Dict] = []
        self._goal         = ""
        self._last_call    = 0.0
        self._loaded       = False

    def load(self):
        """Initialise Groq client."""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                logger.error("GROQ_API_KEY not set in .env")
                return
            self._client = Groq(api_key=api_key)
            self._loaded = True
            logger.info("Groq LLM agent loaded.")
        except ImportError:
            logger.error("groq not installed. Run: pip install groq")
        except Exception as e:
            logger.error(f"Groq init error: {e}")

    def set_goal(self, goal_text: str):
        """Set a new user goal. Resets conversation history."""
        self._goal    = goal_text.strip()
        self._history = []
        logger.info(f"Agent goal set: '{self._goal}'")

    def reason(self, scene_description: str) -> Optional[Dict[str, Any]]:
        """
        Main reasoning call. Takes scene description, returns agent decision.

        Args:
            scene_description: output of quant.build_scene_description()

        Returns:
            Parsed JSON dict or None on failure.
        """
        if not self._loaded or self._client is None:
            return self._fallback_response(scene_description)

        # Rate limit — don't hammer the API
        elapsed = time.time() - self._last_call
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)

        user_msg = (
            f"Current goal: {self._goal}\n\n"
            f"Scene: {scene_description}"
        )

        self._history.append({"role": "user", "content": user_msg})

        # Trim history to MAX_HISTORY turns
        if len(self._history) > self.MAX_HISTORY * 2:
            self._history = self._history[-self.MAX_HISTORY * 2:]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        for attempt in range(self.RETRY_LIMIT):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                self._last_call = time.time()
                raw = response.choices[0].message.content.strip()
                parsed = self._parse_response(raw)

                if parsed:
                    self._history.append({"role": "assistant", "content": raw})
                    return parsed

            except Exception as e:
                logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))

        return self._fallback_response(scene_description)

    def _parse_response(self, raw: str) -> Optional[Dict]:
        """Parse and validate LLM JSON response."""
        try:
            # Strip any accidental markdown fences
            clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
            data = json.loads(clean)

            # Validate required fields
            required = ("speak", "detect_next", "action", "goal_reached")
            if not all(k in data for k in required):
                logger.warning(f"Missing fields in LLM response: {data.keys()}")
                return None

            # Sanitise
            data["speak"]       = str(data["speak"])[:300]
            data["detect_next"] = [str(l) for l in data["detect_next"][:8]]
            data["action"]      = str(data["action"])
            data["goal_reached"] = bool(data.get("goal_reached", False))
            return data

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM response parse error: {e} — raw: {raw[:100]}")
            return None

    def _fallback_response(self, scene: str) -> Dict:
        """
        Offline fallback when Groq is unavailable.
        Uses simple rule-based logic to still be useful.
        """
        speak = "Moving forward carefully. Listening for voice commands."
        action = "move_forward"

        # Simple obstacle detection from scene string
        if "caution" in scene.lower() or "very close" in scene.lower():
            speak  = "Obstacle detected very close. Please stop and wait."
            action = "stop"
        elif "left" in scene.lower() and "0." in scene:
            speak  = "Object on your left. Moving slightly right."
            action = "turn_right"
        elif "right" in scene.lower() and "0." in scene:
            speak  = "Object on your right. Moving slightly left."
            action = "turn_left"
        elif self._goal:
            speak = f"Searching for {self._goal}. Moving forward slowly."

        return {
            "speak":        speak,
            "detect_next":  self._default_labels(),
            "action":       action,
            "goal_reached": False,
            "reasoning":    "Offline fallback — Groq unavailable",
        }

    def _default_labels(self) -> List[str]:
        """Default detection labels when goal is known."""
        base = ["door", "chair", "person", "obstacle", "steps", "table"]
        if self._goal:
            goal_word = self._goal.split()[0].lower()
            if goal_word not in base:
                base.insert(0, goal_word)
        return base[:6]

    def get_goal(self) -> str:
        return self._goal
