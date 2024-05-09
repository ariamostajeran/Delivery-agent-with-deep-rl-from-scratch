"""Value Iteration Agent.

This is an agent that takes the action with the highest expected return in that state.
"""

from random import randint
import numpy as np

# Added
import math
import copy

from agents import BaseAgent


class ValueAgent(BaseAgent):
    def __init__(self, stateSpace, actionSpace, gamma, cols, P, R):
        self.stateSpace = stateSpace
        self.actionSpace = actionSpace
        self.gamma = gamma
        self.V = [0.0 for _ in range(len(self.stateSpace))]
        self.pi = [0 for _ in range(len(self.stateSpace))]
        self.cols = cols
        self.P = P
        self.R = R

    def pointer(self, state: tuple[int, int]):
        return state[0] + state[1] * self.cols

    def update(self, stateSpace):
        # Policy evaluation
        for state in stateSpace:
            best_value = -math.inf
            for action in self.actionSpace:
                expected_value = (self.R[action][self.pointer(state)] 
                                    + self.gamma*sum(
                                        self.P[action][self.pointer(state)][self.pointer(next_state)] 
                                        * self.V[self.pointer(next_state)] 
                                            for next_state in self.stateSpace))
                if expected_value > best_value:
                    best_value = copy.copy(expected_value)
            self.V[self.pointer(state)] = best_value

        # Policy control
        for state in stateSpace:
            best_value, best_action = -math.inf, None
            for action in self.actionSpace:
                expected_value = (self.R[action][self.pointer(state)] 
                                    + self.gamma*sum(
                                        self.P[action][self.pointer(state)][self.pointer(next_state)] 
                                        * self.V[self.pointer(next_state)] 
                                            for next_state in self.stateSpace))
                if expected_value > best_value:
                    best_value = copy.copy(expected_value)
                    best_action = copy.copy(action)
            self.pi[self.pointer(state)] = best_action

    def take_action(self, state: tuple[int, int]) -> int:
        # Look up the best action to take for the given state in the policy
        return self.pi[self.pointer(state)]