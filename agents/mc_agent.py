"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np

from agents import BaseAgent


class MonteCarloAgent(BaseAgent):

    def __init__(self, num_states, num_actions, grid_width, gamma=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.q_values = np.zeros((num_states, num_actions))
        self.n_visits = np.zeros((num_states, num_actions))
        self.grid_width = grid_width

    def update(self, state_action_reward_list):
        """Updates the agent's q-values after each round of exploration"""
        G_value = 0
        # Iterate backwards through list
        for state, action, reward in reversed(state_action_reward_list):
            state_index = self.encode_state(state)

            # Increment the amount of times this state-action pair has been played
            self.n_visits[state_index, action] += 1
            
            # Updating q-values G = (gamma*G) + R_(t+1)
            self.q_values[state_index, action] += self.gamma*G_value + reward
            
            # Take average over number of visits to this state-action pair (Monte Carlo method)
            self.q_values[state_index, action] /= self.n_visits[state_index, action]

            G_value = reward

    def encode_state(self, state):
        return state[0] * self.grid_width + state[1]

    def take_action(self, state: tuple[int, int]) -> int:
        state_index = self.encode_state(state)
        return np.argmax(self.q_values[state_index])
    
    def take_random_action(self, state: tuple[int, int]) -> int:
        return randint(0, 3)