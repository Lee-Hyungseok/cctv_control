from cctv_environment import CCTVEnvironment, Direction
from typing import Dict

class BaselineCCTV:
    def __init__(self):
        self.current_direction_index = 0
        self.directions = list(Direction)
        # Sequential scanning: rotate clockwise every step
        # 1 episode = 144 steps → each direction monitored 36 times (144/4)

    def choose_action(self, state: Dict) -> int:
        # Rotate clockwise every step: NORTH → SOUTH → EAST → WEST → NORTH ...
        action = self.current_direction_index
        self.current_direction_index = (self.current_direction_index + 1) % len(self.directions)
        return action

    def reset(self):
        self.current_direction_index = 0