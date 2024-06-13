import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from agents.double_dqn_agent import DoubleDQNAgent #Currently being worked on
from value_agent_functions import *
try:
    from world import Environment
    from agents.random_agent import RandomAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment

#Plots the results of the experiments, unchanged from A1
def plot_experiment(data, xlabel, ylabel, title):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_v_matrix(agent, grid_shape, agent_name):
    """Plot the V matrix as a heatmap."""

    if agent_name == "DoubleDQNAgent":
        Q = np.array(agent.q_values)
        Q = Q.reshape((grid_shape[1], grid_shape[0], 4))

        # Create a colormap that treats np.nan values as black
        cmap = cm.viridis
        cmap.set_bad(color='black')

        # Calculate V matrix
        V = np.max(Q, axis=2)

    # Plot V matrix
    plt.imshow(V, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(f'V-Matrix Heatmap for {agent_name}')
    plt.show()

def hyperparameter_search(env: Environment, random_seed: int, agent_name: str, iters: int, sigma: float):
    """Perform a hyperparameter search over a range of values."""

    # Define the range of hyperparameters to test
    carrying_capacity = [1,3,5]

    # Set default values. Determined to be optimal in A1. Kept as a constant to test only for carrying capacity
    default_epsilon = 0.05
    default_epsilon_decay = 1
    default_gamma = 0.95
    default_alpha = 0.4
    default_epsilon_start = 0.5
    default_epsilon_end = 0.01
    default_decay_steps = 10**5 #Was not discovered during A1. Test if this is a suitable value

    # Manually set x-ticks to scale to number of iterations
    x_vals = np.arange(0, iters, iters/10)

    if agent_name == "DoubleDQNAgent":
        plt.figure(figsize=(10, 7))
        cum_rewards = []
        for capacity in carrying_capacity:
            agent = DoubleDQNAgent(env,
                                   decay_steps=default_decay_steps,
                                   gamma=default_gamma,
                                   device=None, #NOTE: This parameter must be changed to a suitable value
                                   start_epsilon=default_epsilon_start,
                                   end_epsilon=default_epsilon_end,
                                   carrying_capacity=capacity, #NOTE: This is currently not yet a parameter
                                   n_actions=4,
                                   batch_size=128)
            cum_rewards_gamma = agent.train(iters)
            
            cum_rewards.append(cum_rewards_gamma)
        # Plot cumulative rewards over iterations for each gamma value
        for i, capacity in enumerate(carrying_capacity):
            plt.plot(x_vals, cum_rewards[i], label=f'capacity={capacity}')

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Average Cumulative Rewards per gamma-value for {agent_name} agent')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        raise ValueError(f"Agent name doesn't exist")

def plot_all_grid_rewards(all_rewards, grid_paths, xlabel, ylabel, title, iters):
    plt.figure(figsize=(10, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    x_vals = np.arange(0, iters, iters/10)

    for rewards, grid in zip(all_rewards, grid_paths):
        grid_name = str(grid).split("/")[1].split(".")[0]
        plt.plot(x_vals, rewards, label=grid_name)
    
    plt.legend()
    plt.show()