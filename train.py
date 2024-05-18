"""
Train your RL Agent in this file. 
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
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
    from agents.random_agent import RandomAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--agent", type=str, default='qlearning',
                   help="Agent to run with")

    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    p.add_argument("--alpha", type=float, default=None, help="Learning rate.")
    p.add_argument("--epsilon", type=float, default=None, help="Exploration rate.")
    p.add_argument("--plot_rewards", action="store_true", help="Plot cumulative rewards")
    p.add_argument("--vis_matrix", action="store_true", help="Visualize V or Q matrix")
    p.add_argument("--hyperparameters_tuning", action="store_true", help="Perform hyperparameters tuning")
    p.add_argument("--expl_tradeoff", action="store_true", help="Plot exploration/exploitation trade-off")
    return p.parse_args()


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

    if agent_name == "value":
        V = agent.V
        V = [V[i] if V[i] > -9999 else np.nan for i in range(len(V))]
        
        V = np.array(V).reshape((grid_shape[1], grid_shape[0]))

        # Create a colormap that treats np.nan values as black
        cmap = cm.viridis
        cmap.set_bad(color='black')

    elif agent_name == "qlearning" or agent_name == "mc":
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
    plt.title('V Matrix Heatmap')
    plt.show()



def reward_fn(grid, agent_pos) -> float:

    match grid[agent_pos]:
        case 0:  # Moved to an empty tile
            reward = -0.1
        case 1 | 2:  # Moved to a wall or obstacle
            reward = -1
        case 3:  # Moved to a target tile
            reward = 10
            # "Illegal move"
        case _:
            raise ValueError(f"Grid cell should not have value: {grid[agent_pos]}.",
                             f"at position {agent_pos}")
    return reward


def hyperparameter_search(env: Environment, random_seed: int, agent_name: str, iters: int):
    """Perform a hyperparameter search over a range of values."""

    # Define the range of hyperparameters to test
    gamma_values = [0.9, 0.95, 0.99]
    alpha_values = [0.1, 0.2, 0.3]
    epsilon_values = [0.1, 0.2, 0.3]

    # Set default values
    default_gamma = 0.9
    default_alpha = 0.1
    default_epsilon = 0.1


    if agent_name == "qlearning":
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
            cum_rewards_gamma = agent.train(iters)[0]
            # Plot cum rewards over iterations
            cum_rewards.append(cum_rewards_gamma)
        
        # Plot cumulative rewards over iterations for each gamma value
        for i, gamma in enumerate(gamma_values):
            plt.plot(cum_rewards[i], label=f'Gamma={gamma}')

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title('Average Cumulative Rewards Over Training Iterations')
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
            cum_rewards_alpha = agent.train(iters)[0]
            # Plot cum rewards over iterations
            cum_rewards.append(cum_rewards_alpha)
        
        # Plot cumulative rewards over iterations for each alpha value
        for i, alpha in enumerate(alpha_values):
            plt.plot(cum_rewards[i], label=f'Alpha={alpha}')
        
        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title('Average Cumulative Rewards Over Training Iterations')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        cum_rewards = []
        for epsilon in epsilon_values:
            cum_rewards_epsilon = []
            agent = QLearningAgent(env, 
                            num_actions=len(range(4)),
                            alpha=default_alpha,
                            gamma=default_gamma,
                            epsilon=epsilon,
                            random_seed=random_seed)
            cum_rewards_epsilon = agent.train(iters)[0]
            # Plot cum rewards over iterations
            cum_rewards.append(cum_rewards_epsilon)
        
        # Plot cumulative rewards over iterations for each epsilon value
        for i, epsilon in enumerate(epsilon_values):
            plt.plot(cum_rewards[i], label=f'Epsilon={epsilon}')

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title('Average Cumulative Rewards Over Training Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    elif agent_name == "value":
        plt.figure(figsize=(10, 7))
        cum_rewards = []
        for gamma in gamma_values:
            agent = ValueAgent(env, 
                        state_space=make_states(env.grid.shape[0], env.grid.shape[1]), 
                        action_space=range(4),
                        gamma=gamma,
                        random_seed=random_seed)
            cum_rewards_gamma = agent.train(iters)[0]
            cum_rewards.append(cum_rewards_gamma)

        # Plot cumulative rewards over iterations for each gamma value
        for i, gamma in enumerate(gamma_values):
            plt.plot(cum_rewards[i], label=f'Gamma={gamma}')

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title('Average Cumulative Rewards Over Training Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()
       
    elif agent_name == "mc":
        plt.figure(figsize=(10, 7))
        cum_rewards = []
        for gamma in gamma_values:
            agent = MonteCarloAgent(env,
                        num_actions=4,
                        gamma=gamma,
                        random_seed=random_seed)
            cum_rewards_gamma = agent.train(iters)[0]
            cum_rewards.append(cum_rewards_gamma)
        
        # Plot cumulative rewards over iterations for each gamma value
        for i, gamma in enumerate(gamma_values):
            plt.plot(cum_rewards[i], label=f'Gamma={gamma}')

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.title('Average Cumulative Rewards Over Training Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    else:
        raise ValueError(f"Agent name doesn't exists")



def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_name: str, gamma: float, alpha: float = None, epsilon: float = None, plot_rewards: bool = False, vis_matrix: bool = False, hyperparameters_tuning: bool = False, expl_tradeoff: bool = False):
    """Main loop of the program."""

    #Hyperparameters
    alpha = alpha if alpha is not None else 0.1
    gamma = gamma
    epsilon = epsilon if epsilon is not None else 0.1

    for grid in grid_paths:
        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed, reward_fn=reward_fn)

        cum_rewards = []
        if agent_name == "qlearning":
            agent = QLearningAgent(env, 
                            num_actions=len(range(4)),
                            alpha=alpha,
                            gamma=gamma,
                            epsilon=epsilon,
                            random_seed=random_seed)        
        elif agent_name == "value":
            agent = ValueAgent(env, 
                       state_space=make_states(env.grid.shape[0], env.grid.shape[1]), 
                       action_space=range(4),
                       gamma=gamma,
                       random_seed=random_seed)        
        elif agent_name == "mc":
            agent = MonteCarloAgent(env,
                            num_actions=4,
                            gamma=gamma,
                            random_seed=random_seed)
        else:
            raise ValueError(f"Agent name doesn't exists")


        results = agent.train(iters)

        if plot_rewards:
            #Training        
            cum_rewards = results[0]
            plot_experiment(cum_rewards, 'Iterations', 'Cumulative Reward', 'Average Cumulative Rewards Over Training Iterations')
        
        if vis_matrix:
            print("Visualizing V matrix")
            plot_v_matrix(agent, env.grid.shape, agent_name)

        if hyperparameters_tuning:
            hyperparameter_search(env, random_seed, agent_name, iters)

        if expl_tradeoff and len(results) > 1:
            plot_experiment(results[1], 'Iterations', 'Trade-off', 'Exploration/Exploitation trade-off')

        Environment.evaluate_agent(grid_fp=grid, agent=agent, max_steps=iters, sigma=env.sigma, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent, args.gamma, args.alpha, args.epsilon, args.plot_rewards, args.vis_matrix, args.hyperparameters_tuning, args.expl_tradeoff)
