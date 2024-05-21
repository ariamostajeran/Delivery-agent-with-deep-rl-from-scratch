import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from agents.qlearning_agent import QLearningAgent
from agents.value_agent import ValueAgent
from value_agent_functions import *
from agents.mc_agent import MonteCarloAgent
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

    if agent_name == "Value Iteration":
        V = agent.V
        V = [V[i] if V[i] > -9999 else np.nan for i in range(len(V))]
        
        V = np.array(V).reshape((grid_shape[1], grid_shape[0]))

        # Create a colormap that treats np.nan values as black
        cmap = cm.viridis
        cmap.set_bad(color='black')

    elif agent_name == "Q-Learning" or agent_name == "Monte Carlo":
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
    gamma_values = [0.5, 0.8, 0.95]
    alpha_values = [0.1, 0.2, 0.3]

    epsilon_values = [0.1, 0.2, 0.05, 1, 0.5]
    epsilon_decay_values = [1, 1, 1, 0.995, 0.999]
    default_epsilon = 0.1

    # Set default values
    default_gamma = 0.9
    default_alpha = 0.1

    # Manually set x-ticks to scale to number of iterations
    x_vals = np.arange(0, iters, iters/10)


    if agent_name == "Q-Learning":
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        cum_rewards = []
        for gamma in gamma_values:
            cum_rewards_gamma = []
            agent = QLearningAgent(env,
                            num_actions=len(range(4)),
                            alpha=default_alpha,
                            gamma=gamma,
                            epsilon=default_epsilon,
                            random_seed=random_seed)
            cum_rewards_gamma = agent.train(iters)
            # Plot cum rewards over iterations
            cum_rewards.append(cum_rewards_gamma)
        
        # Plot cumulative rewards over iterations for each gamma value
        for i, gamma in enumerate(gamma_values):
            plt.plot(x_vals, cum_rewards[i], label=f'Gamma={gamma}')

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Average Cumulative Rewards per gamma-value for {agent_name} agent')
        plt.legend()
        plt.grid(True)
        plt.subplot(3, 1, 2)

        cum_rewards = []
        for alpha in alpha_values:
            cum_rewards_alpha = []
            agent = QLearningAgent(env,
                            num_actions=len(range(4)),
                            alpha=alpha,
                            gamma=default_gamma,
                            epsilon=default_epsilon,
                            random_seed=random_seed)
            cum_rewards_alpha = agent.train(iters)
            # Plot cum rewards over iterations
            cum_rewards.append(cum_rewards_alpha)
        
        # Plot cumulative rewards over iterations for each alpha value
        for i, alpha in enumerate(alpha_values):
            plt.plot(x_vals, cum_rewards[i], label=f'Alpha={alpha}')

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Average Cumulative Rewards per alpha-value for {agent_name} agent')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        cum_rewards = []
        for i, epsilon in enumerate(epsilon_values):
            cum_rewards_epsilon = []
            agent = QLearningAgent(env, 
                            num_actions=len(range(4)),
                            alpha=default_alpha,
                            gamma=default_gamma,
                            epsilon=epsilon,
                            decay= epsilon_decay_values[i],
                            min_epsilon=0.01,
                            random_seed=random_seed)
            cum_rewards_epsilon = agent.train(iters)
            # Plot cum rewards over iterations
            cum_rewards.append(cum_rewards_epsilon)
        
        # Plot cumulative rewards over iterations for each epsilon value
        for i, epsilon in enumerate(epsilon_values):
            plt.plot(x_vals, cum_rewards[i], label=f'Epsilon={epsilon}\n Decay={epsilon_decay_values[i]}')

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Average Cumulative Rewards per epsilon-value for {agent_name} agent')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    elif agent_name == "Value Iteration":
        plt.figure(figsize=(10, 7))
        cum_rewards = []
        for gamma in gamma_values:
            agent = ValueAgent(env, 
                        state_space=make_states(env.grid.shape[0], env.grid.shape[1]), 
                        action_space=range(4),
                        gamma=gamma,
                        sigma=sigma,
                        random_seed=random_seed,)
            cum_rewards_gamma = agent.train(iters)
            cum_rewards.append(cum_rewards_gamma)

        # Plot cumulative rewards over iterations for each gamma value
        for i, gamma in enumerate(gamma_values):
            plt.plot(x_vals, cum_rewards[i], label=f'Gamma={gamma}')

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Average Cumulative Rewards per gamma-value for {agent_name} agent')
        plt.legend()
        plt.grid(True)
        plt.show()
       
    elif agent_name == "Monte Carlo":
        plt.figure(figsize=(10, 7))
        cum_rewards = []
        for gamma in gamma_values:
            agent = MonteCarloAgent(env,
                        num_actions=4,
                        gamma=gamma,
                        random_seed=random_seed)
            cum_rewards_gamma = agent.train(iters)
            
            cum_rewards.append(cum_rewards_gamma)
        # Plot cumulative rewards over iterations for each gamma value
        for i, gamma in enumerate(gamma_values):
            plt.plot(x_vals, cum_rewards[i], label=f'Gamma={gamma}')

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



