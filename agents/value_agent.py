"""Value Iteration Agent.

This is an agent that takes the action with the highest expected return in that state.
"""

from random import randint
import numpy as np

from agents import BaseAgent


class ValueAgent(BaseAgent):
    def __init__(self, stateSpace, actionSpace, gamma, cols, P):
        self.states = stateSpace
        self.actions = actionSpace
        self.gamma = gamma
        self.V = [0.0 for _ in range(len(self.states))]
        self.pi = [0 for _ in range(len(self.actions))]
        self.cols = cols
        self.P = P

    def pointer(self, state: tuple[int, int]):
        return state[0] * self.cols + state[1]

    def update(self, state: tuple[int, int], reward: float, action):
        expected_value = 0
        # Get the transition probabilities for the current state and action
        transition_probs = self.P[(state, action)]
        for next_state, prob in transition_probs.items():
            expected_value += prob * self.V[self.pointer(next_state)]

        self.V[self.pointer(state)] = reward + self.gamma * expected_value

    def take_action(self, state: tuple[int, int]) -> int:
        max_value = float("-inf")
        best_action = None

        for action in self.actions:
            expected_value = 0
            # Get the transition probabilities for the current state and action
            transition_probs = self.P[(state, action)]
            for next_state, prob in transition_probs.items():
                expected_value += prob * self.V[self.pointer(next_state)]

            if expected_value > max_value:
                max_value = expected_value
                best_action = action

        return best_action
