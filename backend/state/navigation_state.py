import time


class NavigationState:
    VALID_STATES = {
        "IDLE",
        "SEARCH",
        "ALIGN",
        "ADVANCE",
        "AVOID",
        "STOP",
        "BLIND",
    }

    def __init__(self):
        self.current_state = "IDLE"
        self.previous_state = None

        self.state_enter_time = time.time()

        # context signals
        self.target_locked = False
        self.collision_active = False

        # behavioral memory
        self.last_action = None
        self.goal_reached = False

    def transition(self, new_state):
        if new_state not in self.VALID_STATES:
            raise ValueError(f"Invalid state: {new_state}")

        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_enter_time = time.time()

    def time_in_state(self):
        return time.time() - self.state_enter_time

    def set_collision(self, flag: bool):
        self.collision_active = flag

    def set_target_lock(self, flag: bool):
        self.target_locked = flag

    def snapshot(self):
        return {
            "state": self.current_state,
            "previous_state": self.previous_state,
            "time_in_state": self.time_in_state(),
            "collision": self.collision_active,
            "target_locked": self.target_locked,
            "goal_reached": self.goal_reached,
        }