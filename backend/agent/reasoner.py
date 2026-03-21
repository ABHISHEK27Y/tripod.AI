import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class AgentReasoner:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            self.model = "llama-3.3-70b-versatile" # Fast reasoning model
            print("[Reasoner] Groq API initialized.")
        else:
            print("[Warning] GROQ_API_KEY not set. Reasoner will run in fallback dummy mode.")
            self.client = None

    def decide_next_target(self, user_goal, world_snapshot):
        if not self.client:
            # Fallback dummy logic
            target = "chair"
            if "door" in user_goal.lower(): target = "door"
            elif "toilet" in user_goal.lower(): target = "toilet"
            elif "chair" in user_goal.lower(): target = "chair"
            
            return {
                "sub_goal": target, 
                "spoken_guidance": f"I'm looking for a {target}.", 
                "status": "IN_PROGRESS"
            }
            
        system_prompt = "You are the reasoning engine for a visually impaired person's navigation assistant. Respond ONLY in valid JSON format."
        user_prompt = f"""
The user's spoken high-level goal is: "{user_goal}"

The current camera snapshot sees these objects (with their distances and directions):
{json.dumps(world_snapshot['objects'])}

Your goal is to decide the next immediate sub-goal object to find. 
Valid objects depend on what a YOLOv8 standard COCO model can detect, or common obstacles (e.g. 'chair', 'person', 'bed', 'toilet', 'sink', 'tv', 'door').
If the user's goal object is not visible, guide them to something intermediate (like a 'door' to leave the room).
If the goal is visible, make it the sub_goal and provide encouraging spoken guidance.

Respond ONLY with JSON containing:
- "sub_goal": string (a class name to target, e.g., "door", "toilet", "chair")
- "spoken_guidance": string (a short sentence to speak to the user, like "I see the door ahead, let's keep moving.")
- "status": string ("IN_PROGRESS", "COMPLETED", or "IMPOSSIBLE")
"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            response_text = chat_completion.choices[0].message.content
            data = json.loads(response_text)
            return data
        except Exception as e:
            print(f"[Reasoner] LLM failed: {e}")
            # Fallback in case of error
            return {"sub_goal": "door", "spoken_guidance": "Thinking...", "status": "IN_PROGRESS"}
