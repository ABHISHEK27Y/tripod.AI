class NavigatorFSM:
    def __init__(self, target_label="door"):
        self.target_label = target_label

    def decide(self, world, nav_state):
        """
        world = world_state.snapshot()
        nav_state = NavigationState instance
        """

        # ---------- NO VISION / BLIND STATE ----------
        # If no objects are detected, we have no vision (camera covered, fog, etc)
        # This is a critical safety state - stop and don't move
        if len(world["objects"]) == 0:
            nav_state.transition("BLIND")
            return {
                "action": "STOP",
                "reason": "No vision - camera obstructed or no objects detected",
                "state": "BLIND",
            }

        # ---------- SAFETY OVERRIDE ----------
        if world["collision_risk"]:
            nav_state.set_collision(True)
            nav_state.transition("AVOID")
        else:
            nav_state.set_collision(False)

        # ---------- TARGET VISIBILITY ----------
        if world["target_visible"]:
            nav_state.set_target_lock(True)
        else:
            nav_state.set_target_lock(False)

        state = nav_state.current_state

        # ---------- FSM TRANSITIONS ----------
        if state == "IDLE":
            nav_state.transition("SEARCH")

        elif state == "SEARCH":
            if nav_state.collision_active:
                nav_state.transition("AVOID")
            elif nav_state.target_locked:
                nav_state.transition("ALIGN")

        elif state == "ALIGN":
            if nav_state.collision_active:
                nav_state.transition("AVOID")
            elif not nav_state.target_locked:
                nav_state.transition("SEARCH")
            elif self._target_centered(world):
                nav_state.transition("ADVANCE")

        elif state == "ADVANCE":
            if nav_state.collision_active:
                nav_state.transition("AVOID")
            elif not nav_state.target_locked:
                nav_state.transition("SEARCH")

        elif state == "AVOID":
            if not nav_state.collision_active:
                # return to previous meaningful state
                if nav_state.target_locked:
                    nav_state.transition("ALIGN")
                else:
                    nav_state.transition("SEARCH")

        elif state == "STOP":
            if not nav_state.collision_active:
                nav_state.transition("SEARCH")

        # ---------- ACTION OUTPUT ----------
        return self._action_for_state(world, nav_state)

    def _target_centered(self, world):
        for o in world["objects"]:
            if o["label"] == self.target_label and o["direction"] == "center":
                return True
        return False

    def _action_for_state(self, world, nav_state):
        state = nav_state.current_state

        if state == "SEARCH":
            return {
                "action": self._best_clear_direction(world),
                "reason": "Searching safe path",
                "state": state,
            }

        if state == "ALIGN":
            return {
                "action": self._target_direction(world),
                "reason": "Aligning to target",
                "state": state,
            }

        if state == "ADVANCE":
            return {
                "action": "FORWARD",
                "reason": "Advancing to target",
                "state": state,
            }

        if state == "AVOID":
            return {
                "action": self._avoid_direction(world),
                "reason": "Obstacle avoidance",
                "state": state,
            }

        if state == "STOP":
            return {
                "action": "STOP",
                "reason": "Safety halt",
                "state": state,
            }

        return {
            "action": "SEARCH",
            "reason": "Fallback",
            "state": state,
        }

    # ---------- HELPER POLICIES ----------

    def _best_clear_direction(self, world):
        rank = {"near": 0, "mid": 1, "far": 2}
        clearance = world["zone_clearance"]

        best = max(clearance, key=lambda z: rank[clearance[z]])

        if best == "left":
            return "MOVE_LEFT"
        elif best == "right":
            return "MOVE_RIGHT"
        else:
            return "FORWARD"

    def _target_direction(self, world):
        for o in world["objects"]:
            if o["label"] == self.target_label:
                if o["direction"] == "left":
                    return "MOVE_LEFT"
                if o["direction"] == "right":
                    return "MOVE_RIGHT"
        return "FORWARD"

    def _avoid_direction(self, world):
        rank = {"near": 0, "mid": 1, "far": 2}
        clearance = world["zone_clearance"]

        # move away from worst zone
        worst = min(clearance, key=lambda z: rank[clearance[z]])

        if worst == "left":
            return "MOVE_RIGHT"
        elif worst == "right":
            return "MOVE_LEFT"
        else:
            return "STOP"