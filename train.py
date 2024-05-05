"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.value_agent import ValueAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.value_agent import ValueAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument(
        "GRID",
        type=Path,
        nargs="+",
        help="Paths to the grid file to use. There can be more than " "one.",
    )
    p.add_argument(
        "--no_gui", action="store_true", help="Disables rendering to train faster"
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Sigma value for the stochasticity of the environment.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second to render at. Only used if " "no_gui is not set.",
    )
    p.add_argument(
        "--iter", type=int, default=1000, help="Number of iterations to go through."
    )
    p.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed value for the environment.",
    )
    return p.parse_args()

def pointer(i, rows):
    return (i % (rows), i // (rows))

# def get_P_matrix(grid, size, actions):
#     # Initialize probability matrix
#     P = [[[0 for _ in range(size)] for _ in range(size)] for _ in range(len(actions))]
#     for action in enumerate(P):
#         for p in range(size):
#             for q in range(size):
#                 c_from = pointer(p, grid.shape[0])
#                 c_to = pointer(q, grid.shape[0])
#                 if action[0] == 0 and c_from[0] == c_to[0] and c_from[1] == c_to[1] - 1 and (grid[c_from[0]][c_from[1]] == 0 or grid[c_from[0]][c_from[1]] == 4 or grid[c_from[0]][c_from[1]] == 3) and (grid[c_to[0]][c_to[1]] == 0 or grid[c_to[0]][c_to[1]] == 4 or grid[c_to[0]][c_to[1]] == 3):
#                     P[action[0]][p][q] = 1
#                 elif action[0] == 1 and c_from[0] == c_to[0] and c_from[1] == c_to[1] + 1 and (grid[c_from[0]][c_from[1]] == 0 or grid[c_from[0]][c_from[1]] == 4 or grid[c_from[0]][c_from[1]] == 3) and (grid[c_to[0]][c_to[1]] == 0 or grid[c_to[0]][c_to[1]] == 4 or grid[c_to[0]][c_to[1]] == 3):
#                     P[action[0]][p][q] = 1
#                 elif action[0] == 2 and c_from[0] == c_to[0] + 1 and c_from[1] == c_to[1] and (grid[c_from[0]][c_from[1]] == 0 or grid[c_from[0]][c_from[1]] == 4 or grid[c_from[0]][c_from[1]] == 3) and (grid[c_to[0]][c_to[1]] == 0 or grid[c_to[0]][c_to[1]] == 4 or grid[c_to[0]][c_to[1]] == 3):
#                     P[action[0]][p][q] = 1
#                 elif action[0] == 3 and c_from[0] == c_to[0] - 1 and c_from[1] == c_to[1] and (grid[c_from[0]][c_from[1]] == 0 or grid[c_from[0]][c_from[1]] == 4 or grid[c_from[0]][c_from[1]] == 3) and (grid[c_to[0]][c_to[1]] == 0 or grid[c_to[0]][c_to[1]] == 4 or grid[c_to[0]][c_to[1]] == 3):
#                     P[action[0]][p][q] = 1
#     return P

def moved_down(c_from, c_to):
    return c_from[0] == c_to[0] and c_from[1] == c_to[1] - 1

def moved_up(c_from, c_to):
    return c_from[0] == c_to[0] and c_from[1] == c_to[1] + 1

def moved_left(c_from, c_to):
    return c_from[0] == c_to[0] + 1 and c_from[1] == c_to[1]

def moved_right(c_from, c_to):
    return c_from[0] == c_to[0] - 1 and c_from[1] == c_to[1]

def hit_wall(grid, c_from, c_to):
    return grid[c_from[0]][c_from[1]] == 1 or grid[c_to[0]][c_to[1]] == 1 or grid[c_from[0]][c_from[1]] == 2 or grid[c_to[0]][c_to[1]] == 2

def get_P_matrix(grid, size, actions):
    # Initialize probability matrix
    P = [[[0 for _ in range(size)] for _ in range(size)] for _ in range(len(actions))]
    for action in enumerate(P):
        for p in range(size):
            for q in range(size):
                c_from = pointer(p, grid.shape[0])
                c_to = pointer(q, grid.shape[0])
                if action[0] == 0 and moved_down(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1
                elif action[0] == 1 and moved_up(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1
                elif action[0] == 2 and moved_left(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1
                elif action[0] == 3 and moved_right(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1
    for action in range(4):
        for p in range(size):
            if sum(P[action][p]) == 0:
                P[action][p][p] = 1
    return P

def move(state: tuple[int, int], action: int):
    if action == 0:
        return (state[0], state[1] + 1)
    elif action == 1:
        return (state[0], state[1] - 1)
    elif action == 2:
        return (state[0] - 1, state[1])
    elif action == 3:
        return (state[0] + 1, state[1])

def get_R_matrix(grid, size, actions):
    R = [[0 for _ in range(size)] for _ in range(len(actions))]
    for action in range(len(actions)):
        for p in range(size):
            c_from = pointer(p, grid.shape[0])
            c_to = move(pointer(p, grid.shape[0]), action)
            if grid[c_from[0]][c_from[1]] == 1 or grid[c_from[0]][c_from[1]] == 2:
                R[action][p] = -1000
            elif grid[c_to[0]][c_to[1]] == 1 or grid[c_to[0]][c_to[1]] == 2:
                R[action][p] = -2
            elif grid[c_to[0]][c_to[1]] == 0:
                R[action][p] = -1
            elif grid[c_to[0]][c_to[1]] == 3 or grid[c_to[0]][c_to[1]] == 4:
                R[action][p] = 10
    return R

def make_states(cols: int, rows: int):
    stateSpace = []
    for row in range(rows):
        for col in range(cols):
            stateSpace.append((col, row))
    return stateSpace

def main(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    random_seed: int,
):
    """Main loop of the program."""

    for grid in grid_paths:

        # Set up the environment
        env = Environment(
            grid, no_gui, sigma=sigma, target_fps=fps, random_seed=random_seed
        )

        # Initialize agent
        nr_states = range(env.grid.shape[0] * env.grid.shape[1])
        stateSpace = make_states(env.grid.shape[0], env.grid.shape[1])
        actionSpace = range(4)
        P = get_P_matrix(env.grid, len(nr_states), actionSpace)
        # for p in P[0]:
        #     print(p)
        # for p in P[1]:
        #     print(p)
        R = get_R_matrix(env.grid, len(nr_states), actionSpace)
        agent = ValueAgent(nr_states, stateSpace, actionSpace, 0.9, env.grid.shape[0], P, R)

        # Always reset the environment to initial state
        state = env.reset()
        for _ in trange(iters):

            agent.update(stateSpace)
            action = agent.take_action(state)
            state, _, terminated, _ = env.step(action)
            if terminated:
                env.reset()

            # Agent takes an action based on the latest observation and info.
            # action = agent.take_action(state)

            # # The action is performed in the environment
            # state, reward, terminated, info = env.step(action)

            # # If the final state is reached, stop.
            # if terminated:
            #     env.reset()
            #     # break

            # agent.update(state, reward, info["actual_action"])

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


if __name__ == "__main__":
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)
