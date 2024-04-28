"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
from agents.qlearning_agent import QLearningAgent
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
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


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
def print_q_values(q_values):
    print("Q-Values:")
    print("───────────────────────")
    print("State   |   Action Values")
    print("───────────────────────")
    for state_index, q_values_state in enumerate(q_values):
        print(f"  {state_index}    |   ", end="")
        for action_index, q_value in enumerate(q_values_state):
            print(f"{q_value:.2f}   ", end="")
        print()
    print("───────────────────────")

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps,
                          random_seed=random_seed, reward_fn=reward_fn)
        grid_shape = env.grid.shape
        num_states = grid_shape[0] * grid_shape[1]
        max_step = 100
        # Initialize agent
        agent = QLearningAgent(num_states, 4, grid_width=grid_shape[1])

        # Always reset the environment to initial state
        state = env.reset()

        for iter in trange(iters):
            print(" Iteration ", iter)
            for i in range(max_step):

                # Agent takes an action based on the latest observation and info.
                action = agent.take_action(state)

                # The action is performed in the environment
                state, reward, terminated, info, next_state = env.step(action)
                # if reward>0:
                #     print("goal_reached")
                agent.update(state, next_state, reward, info["actual_action"])

                # If the final state is reached, stop.
                if terminated or i == max_step - 1:
                    # print("STEP NUM", i)
                    # print("reward", reward)
                    env.reset()
                    # print_q_values(agent.q_values)
                    break



        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)