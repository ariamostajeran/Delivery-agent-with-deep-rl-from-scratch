import numpy as np
from agents import BaseAgent

from tqdm import trange

class QLearningAgent(BaseAgent):
    def __init__(self, env, num_actions, alpha, gamma, epsilon, random_seed, min_epsilon=0.0001, decay=0):
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
        """Turns state into a number"""
        return state[0] * self.grid_width + state[1]
    
    def update_epsilon(self):
        """Decreases epsilon and keep it above a threshold"""
        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)

    def update(self, state, next_state, reward, action):
        """Update Q matrix such that it comes closer to its true value each iteration"""
        state_index = self.encode_state(state)
        next_state_index = self.encode_state(next_state)

        # Temporal Difference (TD) update rule
        self.q_values[state_index, action] += self.alpha * (reward +
                                                      self.gamma * np.max(self.q_values[next_state_index]) -
                                                      self.q_values[state_index, action])

    def take_action(self, state, **kwargs):
        """Take a random action or the best action based on exploration-exploitation parameter epsilon"""
        if np.random.rand() < self.epsilon:
            # Perform an action with the aim of exploration
            return np.random.choice(range(self.num_actions))
        else:
            # Perform an action with the aim of exploitation
            state_index = self.encode_state(state)
            return np.argmax(self.q_values[state_index])

    def train(self, iters):
        """Train Q Learning agent"""
        
        # Always reset the environment to initial state
        state = self.env.reset()

        # Experiment data
        cum_rewards = []
        monitor_time = iters / 10
        cum_reward = 0

        max_step = self.num_states * 2

        # Start a new episode until we performed 'iters' episodes
        for iteration in trange(iters):
            
            # Perform a new step in this episode with a maximum of 'max_step' steps
            for i in range(max_step):

                # Agent takes an action based on the latest observation and info.
                action = self.take_action(state)

                # The action is performed in the environment
                next_state, reward, terminated, info = self.env.step(action)

                # Update expected Q matrix and move to the next state
                self.update(state, next_state, reward, info["actual_action"])
                state = next_state
                cum_reward += reward

                # Perform another run when target is reached
                if terminated or i == max_step - 1:
                    self.env.reset()
                    break
            
            # Update epsilon
            if iteration %  ((iters * 2/3) / 20) == 0:
                self.update_epsilon()

            if iteration % monitor_time == 0:
                cum_rewards.append(cum_reward / monitor_time) # mean of cum rewards of the past episodes
                cum_reward = 0

        return cum_rewards
