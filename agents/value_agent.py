"""Value Iteration Agent.

This is an agent that takes the action with the highest expected return in that state.
"""
from value_agent_functions import get_P_matrix, get_R_matrix

# Added
import math
import copy

from tqdm import trange

from agents import BaseAgent


class ValueAgent(BaseAgent):
    def __init__(self, env, state_space, action_space, gamma, sigma, random_seed):
        super().__init__(env)
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.sigma = sigma
        self.V = [0.0 for _ in range(len(self.state_space))]
        self.pi = [0 for _ in range(len(self.state_space))]
        self.cols = env.grid.shape[0]
        self.nr_states = range(env.grid.shape[0] * env.grid.shape[1])
        self.max_step = self.num_states * 2
        self.P = get_P_matrix(env.grid, len(self.nr_states), self.action_space, self.sigma, env.grid.shape[0], env.grid.shape[1])
        self.R = get_R_matrix(env.grid, len(self.nr_states), self.action_space, self.sigma)
        self.random_seed = random_seed

    def pointer(self, state: tuple[int, int]):
        return state[0] + state[1] * self.cols

    def update(self):
        V_new = copy.copy(self.V)
        pi_new = copy.copy(self.pi)

        # Policy evaluation
        for state in self.state_space:
            best_value = -math.inf
            for action in self.action_space:
                expected_value = self.R[action][self.pointer(state)] + self.gamma*sum(self.P[action][self.pointer(state)][self.pointer(next_state)] * self.V[self.pointer(next_state)] for next_state in self.state_space)
                if expected_value > best_value:
                    best_value = copy.copy(expected_value)
            V_new[self.pointer(state)] = best_value
        self.V = V_new

        # Policy control
        for state in self.state_space:
            best_value, best_action = -math.inf, None
            for action in self.action_space:
                expected_value = self.R[action][self.pointer(state)] + self.gamma*sum(self.P[action][self.pointer(state)][self.pointer(next_state)] * self.V[self.pointer(next_state)] for next_state in self.state_space)
                if expected_value > best_value:
                    best_value = copy.copy(expected_value)
                    best_action = copy.copy(action)
            pi_new[self.pointer(state)] = best_action
        self.pi = pi_new

    def take_action(self, state: tuple[int, int]) -> int:
        # Look up the best action to take for the given state in the policy
        return self.pi[self.pointer(state)]
    
    def train(self, iters):

        # Always reset the environment to initial state
        state = self.env.reset()

        cum_rewards = []
        monitor_time = iters / 20
        cum_reward = 0

        max_step = self.num_states * 2
    
        for iteration in trange(iters):

            for _ in range(max_step):
            
                # Update expected value matrix and policy
                self.update()
                # Get best action to take based on updated policy
                action = self.take_action(state)
                # Perform the step in the environment
                _, reward, terminated, _, state = self.env.step(action)
                cum_reward += reward
                # Perform another run when target is reached
                if terminated:
                    self.env.reset()
                    break
                
            if iteration % monitor_time == 0:
                cum_rewards.append(cum_reward / monitor_time)  # mean of cum rewards of the past episodes
                cum_reward = 0

        return (cum_rewards,)