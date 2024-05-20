import numpy as np
from agents import BaseAgent
from tqdm import trange


class QLearningAgent(BaseAgent):
    def __init__(self, env, num_actions, alpha, gamma, epsilon, epsilon_decay, min_epsilon, random_seed):
        super().__init__(env)
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_values = np.zeros((self.num_states, num_actions))
        self.grid_width = self.env.grid.shape[1]
        self.random_seed = random_seed
        self.old_pos = None

    def encode_state(self, state):
        return state[0] * self.grid_width + state[1]

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update(self, state, next_state, reward, action):
        state_index = self.encode_state(state)
        next_state_index = self.encode_state(next_state)
        self.q_values[state_index, action] += self.alpha * (reward +
                                                      self.gamma * np.max(self.q_values[next_state_index]) -
                                                      self.q_values[state_index, action])

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.num_actions))
        else:
            state_index = self.encode_state(state)
            return np.argmax(self.q_values[state_index])

    def train(self, iters):
        max_step = self.num_states * 2
        state = self.env.reset()

        cum_rewards = []

        monitor_time = iters / 10
        cum_reward = 0
        for iteration in trange(iters):
            for i in range(max_step):
                action = self.take_action(state)
                self.old_pos = state
                next_state, reward, terminated, info = self.env.step(action)
                self.update(self.old_pos, next_state, reward, info["actual_action"])
                cum_reward += reward
                if terminated or i == max_step - 1:
                    self.env.reset()
                    break
            if iteration % int(iters/100) == 0:
                self.update_epsilon()

            if iteration % monitor_time == 0:
                cum_rewards.append(cum_reward / monitor_time)
                cum_reward = 0

        return cum_rewards


