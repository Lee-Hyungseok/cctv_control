import numpy as np
import random
from typing import Tuple, Dict
from enum import Enum

class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class CCTVEnvironment:
    def __init__(self, crime_probability: float = 0.05):
        self.directions = list(Direction)
        self.n_actions = len(self.directions)
        self.crime_probability = crime_probability
        self.current_crimes = {direction: False for direction in self.directions}
        self.time_step = 0
        self.total_crimes = 0
        self.detected_crimes = 0

    def reset(self) -> Dict:
        self.current_crimes = {direction: False for direction in self.directions}
        self.time_step = 0
        self.total_crimes = 0
        self.detected_crimes = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        selected_direction = self.directions[action]

        # Generate new crimes
        self._generate_crimes()

        # Calculate reward
        reward = 0
        if self.current_crimes[selected_direction]:
            reward = 10  # Positive reward for detecting crime
            self.detected_crimes += 1

        # Clear detected crime
        if self.current_crimes[selected_direction]:
            self.current_crimes[selected_direction] = False

        self.time_step += 1

        # Episode ends after 365 days (365 * 24 * 60 = 525600 minutes)
        done = self.time_step >= 525600

        state = self._get_state()
        info = {
            'total_crimes': self.total_crimes,
            'detected_crimes': self.detected_crimes,
            'detection_rate': self.detected_crimes / max(1, self.total_crimes)
        }

        return state, reward, done, info

    def _generate_crimes(self):
        for direction in self.directions:
            if random.random() < self.crime_probability:
                if not self.current_crimes[direction]:
                    self.current_crimes[direction] = True
                    self.total_crimes += 1

    def _get_state(self) -> Dict:
        # State includes current crime status for each direction
        return {
            'crimes': [int(self.current_crimes[direction]) for direction in self.directions],
            'time_step': self.time_step
        }

    def get_detection_probability(self) -> float:
        if self.total_crimes == 0:
            return 0.0
        return self.detected_crimes / self.total_crimes