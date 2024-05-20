import numpy as np
from agents import BaseAgent

from tqdm import trange

class QLearningAgent(BaseAgent):
    def __init__(self, env, num_actions, alpha, gamma, epsilon, min_epsilon, decay, random_seed):
        super().__init__(env)
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.q_values = np.zeros((self.num_states, num_actions))
        self.grid_width = self.env.grid.shape[1]

        self.random_seed = random_seed

    def encode_state(self, state):
        return state[0] * self.grid_width + state[1]
    def update_epsilon(self):
        self.epsilon = min(self.epsilon*self.decay, self.min_epsilon)
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

    def train(self, iters):
        max_step = self.num_states * 2
        # Always reset the environment to initial state
        state = self.env.reset()

        # Experiments
        cum_rewards = []

        monitor_time = iters / 10
        cum_reward = 0
        for iteration in trange(iters):
            # print(" Iteration ", iter)
            for i in range(max_step):

                # Agent takes an action based on the latest observation and info.
                action = self.take_action(state)

                # The action is performed in the environment
                next_state, reward, terminated, info = self.env.step(action)

                self.update(state, next_state, reward, info["actual_action"])
                state = next_state
                cum_reward += reward
                # If the final state is reached, stop.
                if terminated or i == max_step - 1:
                    self.env.reset()
                    break
            if iteration % monitor_time == 0:
                cum_rewards.append(cum_reward / monitor_time) # mean of cum rewards of the past episodes
                cum_reward = 0

        return cum_rewards
