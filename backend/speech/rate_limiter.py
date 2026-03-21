import time

class SpeechLimiter:
    def __init__(self):
        self.last_action = None
        self.last_time = 0

    def should_speak(self, action):
        now = time.time()

        if action != self.last_action:
            self.last_action = action
            self.last_time = now
            return True

        if now - self.last_time > 4:
            self.last_time = now
            return True

        return False