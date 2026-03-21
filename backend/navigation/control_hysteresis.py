import time


class ControlHysteresis:
    def __init__(
        self,
        collision_hold_time=0.5,
        target_lock_time=0.4,
        action_min_duration=0.6,
    ):
        self.collision_hold_time = collision_hold_time
        self.target_lock_time = target_lock_time
        self.action_min_duration = action_min_duration

        self._collision_start = None
        self._target_start = None
        self._last_action_time = None
        self._last_action = None

    # ---------- collision hysteresis ----------
    def collision_confirmed(self, collision_flag):
        now = time.time()

        if collision_flag:
            if self._collision_start is None:
                self._collision_start = now

            if now - self._collision_start > self.collision_hold_time:
                return True
        else:
            self._collision_start = None

        return False

    # ---------- target lock hysteresis ----------
    def target_confirmed(self, target_flag):
        now = time.time()

        if target_flag:
            if self._target_start is None:
                self._target_start = now

            if now - self._target_start > self.target_lock_time:
                return True
        else:
            self._target_start = None

        return False

    # ---------- action persistence ----------
    def allow_action_change(self, new_action):
        now = time.time()

        if self._last_action is None:
            self._last_action = new_action
            self._last_action_time = now
            return True

        if new_action != self._last_action:
            if now - self._last_action_time > self.action_min_duration:
                self._last_action = new_action
                self._last_action_time = now
                return True
            else:
                return False

        return True

    def get_last_action(self):
        return self._last_action