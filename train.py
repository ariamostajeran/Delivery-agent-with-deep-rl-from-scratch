"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.mc_agent import MonteCarloAgent

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
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--n_episode_steps", type=int, default=50,
                   help="Number of steps to carry out per episode.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, n_episode_steps: int):
    """Main loop of the program."""

    for grid in grid_paths:
        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)
    
        grid_shape = env.grid.shape
        num_states = grid_shape[0] * grid_shape[1]

        # Initialize agent
        agent = MonteCarloAgent(num_states, 4, grid_width = grid_shape[1])

        for _ in trange(iters):
            # Place agent randomly on grid (exploring starts)
            state = env.reset()

            # List to store state-action history and corresponding rewards
            state_action_reward_list = []

            for step in range(n_episode_steps):
                
                # First action will always be random
                if step == 0:
                    action = agent.take_random_action(state)

                # Remaining actions are greedy
                else: 
                    action = agent.take_action(state)

                previous_state = state
                state, reward, terminated, _ = env.step(action)
                state_action_reward_list.append((previous_state, action, reward))

                # If the final state is reached, stop.
                if terminated:
                    break

            agent.update(state_action_reward_list)

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.n_episode_steps)