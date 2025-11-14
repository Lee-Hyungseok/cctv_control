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

        # 각 방향별 범죄 발생률 설정
        # 북쪽: 0.5%, 나머지: 0.1% ~ 0.2%
        self.crime_probabilities = {
            Direction.NORTH: 0.005,  # 0.5%
            Direction.SOUTH: 0.001,  # 0.1%
            Direction.EAST: 0.002,   # 0.2%
            Direction.WEST: 0.0015   # 0.15%
        }

        self.crime_probability = crime_probability  # 하위 호환성을 위해 유지
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

        # Episode ends after 1 day (144 CCTV operations)
        # 실제시간 1일 = 24시간 = 모델상 24분
        # 10분마다 1회 동작 = 24분 동안 144회 동작
        done = self.time_step >= 144

        state = self._get_state()
        info = {
            'total_crimes': self.total_crimes,
            'detected_crimes': self.detected_crimes,
            'detection_rate': self.detected_crimes / max(1, self.total_crimes)
        }

        return state, reward, done, info

    def _generate_crimes(self):
        for direction in self.directions:
            # 각 방향별로 다른 범죄 발생률 적용
            crime_prob = self.crime_probabilities[direction]
            if random.random() < crime_prob:
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