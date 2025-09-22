from cctv_environment import CCTVEnvironment, Direction
from typing import Dict

class BaselineCCTV:
    def __init__(self):
        self.current_direction_index = 0
        self.directions = list(Direction)
        self.minutes_in_direction = 0
        self.switch_interval = 1  # Switch every 1 minute

    def choose_action(self, state: Dict) -> int:
        # Sequential scanning: stay in each direction for 1 minute
        if self.minutes_in_direction >= self.switch_interval:
            self.current_direction_index = (self.current_direction_index + 1) % len(self.directions)
            self.minutes_in_direction = 0

        action = self.current_direction_index
        self.minutes_in_direction += 1
        return action

    def reset(self):
        self.current_direction_index = 0
        self.minutes_in_direction = 0