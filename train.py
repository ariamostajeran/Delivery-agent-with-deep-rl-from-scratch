"""
Train your RL Agent in this file. 
"""
import matplotlib.pyplot as plt
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
    return p.parse_args()


def plot_cum_rewards(cum_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(cum_rewards, label='Cumulative Rewards')
    plt.xlabel('Iterations')
    plt.ylabel('Cumulative Reward')
    plt.title('Average Cumulative Rewards Over Training Iterations')
    plt.legend()
    plt.grid(True)
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


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_name: str):
    """Main loop of the program."""

    #Hyperparameters
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1

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

        #Training        
        cum_rewards = agent.train(iters)

        plot_cum_rewards(cum_rewards)

        Environment.evaluate_agent(grid_fp=grid, agent=agent, max_steps=iters, sigma=env.sigma, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent)