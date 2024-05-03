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
        idx = 1
        # Iterate backwards through list
        for state, action, reward in reversed(state_action_reward_list):

            # G = (gamma*G) + R_(t+1)
            G_value = self.gamma*G_value + reward

            # Only check the first occurence of the state-action pair in the episodes
            if (state, action) not in [(state, action) for state, action, _ in state_action_reward_list[:-idx]]:
                state_index = self.encode_state(state)

                # Increment the amount of observations for this state-action pair 
                self.n_visits[state_index, action] += 1
                
                # Updating q-values 
                self.q_values[state_index, action] += G_value / self.n_visits[state_index, action]
                idx += 1

    def encode_state(self, state):
        return state[0] * self.grid_width + state[1]

    def take_action(self, state: tuple[int, int]) -> int:

        state_index = self.encode_state(state)
        return np.argmax(self.q_values[state_index])
    
    def take_random_action(self, state: tuple[int, int]) -> int:
        return randint(0, 3)