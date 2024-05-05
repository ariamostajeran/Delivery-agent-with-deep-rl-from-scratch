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
    def __init__(self, nr_states, stateSpace, actionSpace, gamma, cols, P, R):
        self.states = nr_states
        self.stateSpace = stateSpace
        self.actions = actionSpace
        self.gamma = gamma
        self.V = [0.0 for _ in range(len(self.states))]
        self.pi = [0 for _ in range(len(self.states))]
        self.cols = cols
        self.rows = int(len(self.states) / cols)
        self.P = P
        self.R = R

    def pointer(self, state: tuple[int, int]):
        # return state[0] + state[1] * self.cols
        # print(state, state[0] + state[1] * self.cols)
        return state[0] + state[1] * self.cols

    def update(self, stateSpace):
        # Policy evaluation
        for state in stateSpace:
            best_value = -math.inf
            for action in range(4):
                expected_value = self.R[action][self.pointer(state)] + self.gamma*sum(self.P[action][self.pointer(state)][self.pointer(next_state)] * self.V[self.pointer(next_state)] for next_state in self.stateSpace)
                if expected_value > best_value:
                    best_value = copy.copy(expected_value)
            self.V[self.pointer(state)] = best_value

        # Policy control
        for state in stateSpace:
            best_value, best_action = -math.inf, None
            for action in range(4):
                expected_value = self.R[action][self.pointer(state)] + self.gamma*sum(self.P[action][self.pointer(state)][self.pointer(next_state)] * self.V[self.pointer(next_state)] for next_state in self.stateSpace)
                if expected_value > best_value:
                    best_value = copy.copy(expected_value)
                    best_action = copy.copy(action)
            self.pi[self.pointer(state)] = best_action

        # Debugging
        # print(self.V)
        # print(self.pi)

        # expected_value = 0

        # next_states = self.states
        # for next_state in next_states:
        #     prob = self.P[action][self.pointer(state)][next_state]
        #     expected_value += prob * self.V[next_state]

        # self.V[self.pointer(state)] = reward + self.gamma * expected_value

    def take_action(self, state: tuple[int, int]) -> int:
        print(self.pi, self.pi[self.pointer(state)], state, self.pointer(state))
        return self.pi[self.pointer(state)]

        # max_value = float("-inf")
        # best_action = None

        # for action in self.actions:
        #     expected_value = 0

        #     next_states = self.states
        #     for next_state in next_states:
        #         prob = self.P[action][self.pointer(state)][next_state]
        #         expected_value += prob * self.V[next_state]

        #     if expected_value > max_value:
        #         max_value = expected_value
        #         best_action = action

        # return best_action