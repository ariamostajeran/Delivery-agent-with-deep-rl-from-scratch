"""
Train your RL Agent in this file. 
"""
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
from agents.qlearning_agent import QLearningAgent
from agents.value_agent import ValueAgent
from value_agent_functions import *
from agents.mc_agent import MonteCarloAgent
from experiments import *
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
    p.add_argument("--compare_grids", action="store_true", help="Perform hyperparameters tuning")
    return p.parse_args()

agent_name_map = {
    "qlearning" : "Q-Learning",
    "value" : "Value Iteration",
    "mc" : "Monte Carlo"
}

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
         sigma: float, random_seed: int, agent_name: str, gamma: float, alpha: float = None, 
         epsilon: float = None, plot_rewards: bool = False, vis_matrix: bool = False, 
         hyperparameters_tuning: bool = False, compare_grids : bool = False):
    """Main loop of the program."""

    #Hyperparameters
    alpha = alpha if alpha is not None else 0.1
    gamma = gamma
    epsilon = epsilon if epsilon is not None else 0.1

    if compare_grids:
        plot_rewards = False
        cum_reward_per_grid = [] 

    for grid in grid_paths:
        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed, reward_fn=reward_fn)

        if agent_name == "qlearning":
            min_epsilon = 0.0001
            decay = 0.95
            init_epsilon = 0.8
            agent = QLearningAgent(env, 
                            num_actions=len(range(4)),
                            alpha=alpha,
                            gamma=gamma,
                            epsilon=init_epsilon,
                            min_epsilon=min_epsilon,
                            decay=decay,
                            random_seed=random_seed)        
        elif agent_name == "value":
            agent = ValueAgent(env, 
                       state_space=make_states(env.grid.shape[0], env.grid.shape[1]), 
                       action_space=range(4),
                       gamma=gamma,
                       sigma=sigma,
                       random_seed=random_seed)        
        elif agent_name == "mc":
            agent = MonteCarloAgent(env,
                            num_actions=4,
                            gamma=gamma,
                            random_seed=random_seed)
        else:
            raise ValueError(f"Agent name doesn't exists")
        
        agent_name_clean = agent_name_map[agent_name]
        results = agent.train(iters)

        if compare_grids:
            cum_reward_per_grid.append(results)

        if plot_rewards:
            #Training        
            cum_rewards = results
            plot_experiment(cum_rewards, 'Iterations', 'Cumulative Reward', 
                            f'Cumulative rewards for {agent_name_clean} agent')
        
        if vis_matrix:
            print("Visualizing V matrix")
            plot_v_matrix(agent, env.grid.shape, agent_name_clean)

        if hyperparameters_tuning:
            hyperparameter_search(env, random_seed, agent_name_clean, iters, sigma)

        Environment.evaluate_agent(grid_fp=grid, agent=agent, max_steps=iters, sigma=env.sigma, random_seed=random_seed)

    if compare_grids:
        plot_all_grid_rewards(cum_reward_per_grid, grid_paths, 'Iterations', 
                              'Cumulative Reward', f'Cumulative rewards per grid for {agent_name_clean} agent', iters)

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, 
         args.random_seed, args.agent, args.gamma, args.alpha, 
         args.epsilon, args.plot_rewards, args.vis_matrix, 
         args.hyperparameters_tuning, args.compare_grids)