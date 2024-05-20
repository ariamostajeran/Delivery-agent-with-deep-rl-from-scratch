
# Reinforcement Learning Assignment

This project contains experiments conducted as part of a reinforcement learning assignment. The experiments involve training agents using various RL algorithms and evaluating their performance on different grid configurations.

## Project Structure

```
├── agents/
│   ├── monte_carlo_agent.py
│   ├── q_learning_agent.py
│   ├── value_iteration_agent.py
├── grid_configs/
│   ├── example_grid.npy
│   ├── large_grid.npy
│   ├── large_maze_grid.npy
│   ├── large_multiple_grid.npy
│   ├── large_sparse_grid.npy
│   ├── medium_maze_grid.npy
│   ├── medium_multiple_grid.npy
│   ├── medium_sparse_grid.npy
│   ├── random_bad.npy
│   ├── random_good.npy
│   ├── small_grid.npy
│   ├── small_maze_grid.npy
│   ├── small_multiple_grid.npy
│   ├── small_sparse_grid.npy
│   ├── solveable.npy
│   ├── test_grid.npy
│   ├── unsolveable.npy
├── plots/
│   ├── Cumulative rewards per grid for Monte Carlo agent.png
│   ├── Cumulative rewards per grid for Q-Learning agent.png
│   ├── Cumulative rewards per grid for Value Iteration agent.png
│   ├── v-matrix.png
├── results/
├── world/
│   ├── __init__.py
│   ├── environment.py
│   ├── grid.py
│   ├── grid_creator.py
│   ├── gui.py
│   ├── helpers.py
│   ├── path_visualizer.py
│   ├── static/
│   ├── templates/
├── train.py
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repository**:
    ```sh
    git clone git@github.com:KGCVX/Assignment-1.git
    cd DIC-2AMC15-2024
    ```

2. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To train the reinforcement learning agents, use the `train.py` script. This script requires certain arguments to specify the agent type, grid configuration, and other parameters.

### Running the Training Script

```sh
python train.py  <grid_config> --agent <agent_type>  [other_options]
```

### Command-Line Options

- `--agent`: Specify the type of agent to train (`mc`, `qlearning`, `value`).
- `GRID`: Paths to the grid file to use. There can be more than one. (e.g., `grid_configs/small_grid.npy`).
- `--iter`: Number of episodes to train the agent (default is 1000).
- `--no_gui`: Disables rendering to train faster
- `--sigma`: Sigma value for the stochasticity of the environment (default 0.1).
- `--fps`: Frames per second to render at. Only used if no_gui is not set (default 30).
- `--random_seed`: Random seed value for the environment (default 0).
- `--gamma`: Discount Factor (default 0.9).
- `--alpha`: Learning Rate (default 0.1).
- `--plot_rewards`: Plot Cumulative Rewards.
- `--vis_matrix`: Visualize V or Q matrix.
- `--hyperparameters_tuning`: Perform hyperparameters tuning.


### Example

```sh
 python train.py .\grid_configs\small_grid.npy --no_gui --plot_rewards --vis_matrix --iter 3000 --sigma 0.3 --agent qlearning --random_seed 100
```

## File Descriptions

- **agents/**: Contains the implementation of different RL agents.
  - `monte_carlo_agent.py`: Implementation of the Monte Carlo agent.
  - `q_learning_agent.py`: Implementation of the Q-Learning agent.
  - `value_iteration_agent.py`: Implementation of the Value Iteration agent.
- **grid_configs/**: Various grid configuration files used for training and testing the agents.
- **plots/**: Contains plots generated from the experiment results.
- **results/**: Stores results from different runs, including both plots and text files.
- **world/**: Contains the environment setup and helper functions for running the simulations.
  - `environment.py`: Defines the environment in which the agents operate.
  - `grid.py`: Manages grid configurations and operations.
  - `grid_creator.py`: Utility for creating new grid configurations.
  - `gui.py`: Handles the graphical user interface for visualizing the environment.
  - `helpers.py`: Helper functions used across the project.
  - `path_visualizer.py`: Visualizes the paths taken by the agents.

