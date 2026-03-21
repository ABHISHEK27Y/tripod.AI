def format_guidance(action, world_state):

    if action == "STOP":
        return "Obstacle very close. Please stop."

    if action == "AVOID_LEFT":
        return "Obstacle ahead. Move slightly left."

    if action == "AVOID_RIGHT":
        return "Obstacle ahead. Move slightly right."

    if action == "FORWARD":
        if world_state.target_visible:
            return "Target ahead. Move forward."
        return "Path is clear. Move forward."

    if action == "SEARCH":
        return "Scanning surroundings. Please wait."

    return None