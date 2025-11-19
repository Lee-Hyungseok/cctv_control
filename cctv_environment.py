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
        self.current_direction = Direction.NORTH  # CCTV가 현재 보고 있는 방향
        self.time_step = 0
        self.total_crimes = 0
        self.detected_crimes = 0
        self.rng = random.Random()  # 독립적인 난수 생성기

        # Pre-generated crime scenario for the episode
        self.crime_scenario = []  # List of sets of directions with crimes at each timestep

    def reset(self, seed: int = None) -> Dict:
        """Reset environment with optional seed for reproducible crime generation"""
        if seed is not None:
            self.rng.seed(seed)

        # Pre-generate entire crime scenario for this episode (144 steps)
        self.crime_scenario = []
        for _ in range(144):
            crimes_at_step = set()
            for direction in self.directions:
                crime_prob = self.crime_probabilities[direction]
                if self.rng.random() < crime_prob:
                    crimes_at_step.add(direction)
            self.crime_scenario.append(crimes_at_step)

        self.current_crimes = {direction: False for direction in self.directions}
        self.current_direction = Direction.NORTH  # 초기 방향
        self.time_step = 0
        # Total crimes is determined by scenario, not by dynamic counting
        self.total_crimes = sum(len(crimes) for crimes in self.crime_scenario)
        self.detected_crimes = 0

        # Direction-wise statistics based on scenario
        self.crimes_by_direction = {
            Direction.NORTH: 0,
            Direction.SOUTH: 0,
            Direction.EAST: 0,
            Direction.WEST: 0
        }
        self.detections_by_direction = {
            Direction.NORTH: 0,
            Direction.SOUTH: 0,
            Direction.EAST: 0,
            Direction.WEST: 0
        }

        # Count crimes by direction from scenario
        for crimes_at_step in self.crime_scenario:
            for direction in crimes_at_step:
                self.crimes_by_direction[direction] += 1

        return self._get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        selected_direction = self.directions[action]

        # CCTV 방향 전환
        self.current_direction = selected_direction

        # Clear all previous crimes (crimes only exist for one timestep)
        self.current_crimes = {direction: False for direction in self.directions}

        # Generate new crimes for this timestep ONLY
        self._generate_crimes()

        # Calculate reward and track detections by direction
        # CCTV는 현재 보고 있는 방향만 탐지 가능
        # 범죄는 이 순간에만 존재하므로, 현재 방향에서 범죄가 발생했는지 확인
        reward = 0
        if self.current_crimes[selected_direction]:
            reward = 10  # Positive reward for detecting crime
            self.detected_crimes += 1
            self.detections_by_direction[selected_direction] += 1

        self.time_step += 1

        # Episode ends after 1 day (144 CCTV operations)
        # 실제시간 1일 = 24시간 = 모델상 24분
        # 10분마다 1회 동작 = 24분 동안 144회 동작
        done = self.time_step >= 144

        state = self._get_state()
        info = {
            'total_crimes': self.total_crimes,
            'detected_crimes': self.detected_crimes,
            'detection_rate': self.detected_crimes / max(1, self.total_crimes),
            'crimes_by_direction': {d.name: self.crimes_by_direction[d] for d in self.directions},
            'detections_by_direction': {d.name: self.detections_by_direction[d] for d in self.directions}
        }

        return state, reward, done, info

    def _generate_crimes(self):
        """Generate crimes based on pre-generated scenario"""
        if self.time_step < len(self.crime_scenario):
            # Use pre-generated crime scenario for this timestep
            crimes_this_step = self.crime_scenario[self.time_step]
            for direction in crimes_this_step:
                # Set crime flag (total_crimes already counted in reset())
                self.current_crimes[direction] = True

    def _get_state(self) -> Dict:
        # State includes ONLY the current viewing direction and whether there's a crime
        # This reflects reality: CCTV can only see one direction at a time
        return {
            'current_direction': self.current_direction.value,
            'crime_detected': int(self.current_crimes[self.current_direction]),
            'time_step': self.time_step
        }

    def get_detection_probability(self) -> float:
        if self.total_crimes == 0:
            return 0.0
        return self.detected_crimes / self.total_crimes