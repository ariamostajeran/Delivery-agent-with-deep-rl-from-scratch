"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np

from agents import BaseAgent

from tqdm import trange


class MonteCarloAgent(BaseAgent):

    def __init__(self, env, num_actions, gamma, random_seed):
        super().__init__(env)
        self.gamma = gamma
        self.num_actions = num_actions
        self.q_values = np.zeros((self.num_states, num_actions))
        self.n_visits = np.zeros((self.num_states, num_actions))
        self.grid_width = self.env.grid.shape[1]
        self.random_seed = random_seed
        self.exploitation_steps = 0
        self.exploration_steps = 0

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
        self.exploitation_steps += 1
        return np.argmax(self.q_values[state_index])

    def take_random_action(self, state: tuple[int, int]) -> int:
        self.exploration_steps += 1
        return randint(0, self.num_actions - 1)
    
    def train(self, iters):
        max_step = self.num_states * 2

        #Experiments
        cum_rewards = []
        expl_tradeoffs = []


        monitor_time = iters / 20
        cum_reward = 0
        for iteration in trange(iters):
            # Place agent randomly on grid (exploring starts)
            state = self.env.reset()

            # List to store state-action history and corresponding rewards
            state_action_reward_list = []

            for step in range(max_step):

                # First action will always be random
                if step == 0:
                    action = self.take_random_action(state)

                # Remaining actions are greedy
                else:
                    action = self.take_action(state)

                previous_state = state
                state, reward, terminated, _, _ = self.env.step(action)
                state_action_reward_list.append((previous_state, action, reward))
                cum_reward += reward

                # If the final state is reached, stop.
                if terminated:
                    break
            if iteration % monitor_time == 0:
                cum_rewards.append(cum_reward / monitor_time)  # mean of cum rewards of the past episodes
                cum_reward = 0
            
                expl_tradeoffs.append(self.exploration_steps/self.exploitation_steps)

            self.update(state_action_reward_list)

        return (cum_rewards, expl_tradeoffs)