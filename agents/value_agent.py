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
        self.V = [0 for _ in range(len(self.states))]
        self.pi = [0 for _ in range(len(self.actions))]
        self.cols = cols
        self.P = P

    def pointer(self, state: tuple[int, int]):
        return state[0]*self.cols + state[1]

    def update(self, state: tuple[int, int], reward: float, action):
        pass

    def take_action(self, state: tuple[int, int]) -> int:
        return