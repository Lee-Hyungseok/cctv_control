import numpy as np
import random
from collections import defaultdict
from typing import Dict, List

class QLearningAgent:
    def __init__(self, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table using defaultdict for dynamic state space
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        # Training metrics
        self.training_rewards = []
        self.training_losses = []
        self.episode_rewards = []

    def _state_to_key(self, state: Dict) -> str:
        # Convert state dict to string key for Q-table
        crimes = tuple(state['crimes'])
        return f"{crimes}"

    def choose_action(self, state: Dict, training: bool = True) -> int:
        state_key = self._state_to_key(state)

        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: best known action
            q_values = self.q_table[state_key]
            return np.argmax(q_values)

    def learn(self, state: Dict, action: int, reward: float,
              next_state: Dict, done: bool):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        current_q = self.q_table[state_key][action]

        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * next_max_q

        # Q-learning update
        td_error = target_q - current_q
        self.q_table[state_key][action] += self.learning_rate * td_error

        # Store metrics
        self.training_rewards.append(reward)
        self.training_losses.append(abs(td_error))

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_table_size(self) -> int:
        return len(self.q_table)

    def get_training_metrics(self) -> Dict:
        return {
            'rewards': self.training_rewards,
            'losses': self.training_losses,
            'episode_rewards': self.episode_rewards,
            'epsilon': self.epsilon,
            'q_table_size': self.get_q_table_size()
        }