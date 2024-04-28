import numpy as np
from agents import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, num_states, num_actions, grid_width, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = np.zeros((num_states, num_actions))
        self.grid_width = grid_width

    def encode_state(self, state):
        return state[0] * self.grid_width + state[1]

    def update(self, state, next_state, reward, action):
        # Q-Learning update rule
        state_index = self.encode_state(state)
        next_state_index = self.encode_state(next_state)
        self.q_values[state_index, action] += self.alpha * (reward +
                                                      self.gamma * np.max(self.q_values[next_state_index]) -
                                                      self.q_values[state_index, action])

    def take_action(self, state, **kwargs):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.num_actions))
        else:
            state_index = self.encode_state(state)
            return np.argmax(self.q_values[state_index])

