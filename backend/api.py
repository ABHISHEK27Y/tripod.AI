from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict

app = FastAPI(title="Agentic AI Control API")

# We use global state to tie the FastAPI app to the running cv2 Agent loop
class AgentState:
    agent_core = None
    navigator = None
    world_state = None

class GoalRequest(BaseModel):
    goal: str

@app.get("/status")
def get_status():
    if not AgentState.agent_core:
        return {"status": "offline", "message": "Agent loop is not running"}
        
    return {
        "status": "online",
        "current_goal": AgentState.agent_core.current_goal,
        "agent_status": AgentState.agent_core.status,
        "fsm_target": AgentState.navigator.target_label if AgentState.navigator else None
    }

@app.post("/goal")
def set_manual_goal(request: GoalRequest):
    """Allows the frontend to manually set a navigation goal (bypassing voice)."""
    if AgentState.agent_core:
        print(f"[API] Manual goal received: {request.goal}")
        AgentState.agent_core.current_goal = request.goal
        AgentState.agent_core.status = "IN_PROGRESS"
        AgentState.agent_core.last_reasoning_time = 0 # Force immediate reasoning
        return {"status": "success", "goal": request.goal}
        
    return {"status": "error", "message": "Agent core not initialized"}
    
@app.get("/world")
def get_world():
    """Returns the current perception world snapshot."""
    if AgentState.agent_core and AgentState.agent_core.world_snapshot:
        return AgentState.agent_core.world_snapshot
    return {"objects": [], "zone_clearance": {}, "collision_risk": False, "target_visible": False}
