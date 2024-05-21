from world.helpers import action_to_direction

def pointer(i, rows):
    """Turns a number into a state (x, y)"""
    return (i % (rows), i // (rows))

def moved_down(c_from, c_to):
    """Checks if the action moves the agent a cell down"""
    return c_from[0] == c_to[0] and c_from[1] == c_to[1] - 1

def moved_up(c_from, c_to):
    """Checks if the action moves the agent a cell up"""
    return c_from[0] == c_to[0] and c_from[1] == c_to[1] + 1

def moved_left(c_from, c_to):
    """Checks if the action moves the agent a cell to the left"""
    return c_from[0] == c_to[0] + 1 and c_from[1] == c_to[1]

def moved_right(c_from, c_to):
    """Checks if the action moves the agent a cell to the right"""
    return c_from[0] == c_to[0] - 1 and c_from[1] == c_to[1]

def hit_wall(grid, c_from, c_to):
    """Checks if the agent is trying to """
    return grid[c_from[0]][c_from[1]] == 1 or grid[c_to[0]][c_to[1]] == 1 or grid[c_from[0]][c_from[1]] == 2 or grid[c_to[0]][c_to[1]] == 2

def remaining_probability(P, p, action, grid, cols, rows, sigma):
    """Computes which states can be reached due to sigma"""
    act = [i for i in range(4)]
    act.remove(action[0])
    for a in act:
        c_from = pointer(p, grid.shape[0])
        c_to = (c_from[0] + action_to_direction(a)[0], c_from[1] + action_to_direction(a)[1])
        if not (c_to[0] < 0 or c_to[1] < 0 or c_to[0] > cols - 1 or c_to[1] > rows - 1):
            if not hit_wall(grid, c_from, c_to):
                P[action[0]][p][c_to[0] + c_to[1] * cols] += sigma/4
            else:
                P[action[0]][p][p] += sigma/4
        else: 
            pass
    return P

def get_P_matrix(grid, size, actions, sigma, cols, rows):
    """Construct the probability matrix"""
    
    # Initialize probability matrix
    P = [[[0 for _ in range(size)] for _ in range(size)] for _ in range(len(actions))]
    for action in enumerate(P):
        for p in range(size):
            c_from = pointer(p, grid.shape[0])
            for q in range(size):
                c_to = pointer(q, grid.shape[0])
                # Once you reach the target you are not supposed to leave it
                if grid[c_from[0]][c_from[1]] == 3:
                    P[action[0]][p][p] = 1
                # Have action to go down, next state is actually down and no obstacle is hit
                elif action[0] == 0 and moved_down(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1-sigma+sigma/4
                    P = remaining_probability(P, p, action, grid, cols, rows, sigma)
                # Have action to go up, next state is actually up and no obstacle is hit
                elif action[0] == 1 and moved_up(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1-sigma+sigma/4
                    P = remaining_probability(P, p, action, grid, cols, rows, sigma)
                # Have action to go left, next state is actually left and no obstacle is hit
                elif action[0] == 2 and moved_left(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1-sigma+sigma/4
                    P = remaining_probability(P, p, action, grid, cols, rows, sigma)
                # Have action to go right, next state is actually right and no obstacle is hit
                elif action[0] == 3 and moved_right(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1-sigma+sigma/4
                    P = remaining_probability(P, p, action, grid, cols, rows, sigma)

    # Else we hit an obstacle, which means that our state does not change
    for action in range(4):
        for p in range(size):
            if sum(P[action][p]) == 0:
                P[action][p][p] = 1
    return P

def move(state: tuple[int, int], action: int):
    """Get new state given a current state and action"""
    if action == 0:
        return (state[0], state[1] + 1)
    elif action == 1:
        return (state[0], state[1] - 1)
    elif action == 2:
        return (state[0] - 1, state[1])
    elif action == 3:
        return (state[0] + 1, state[1])
    
def remaining_reward(action, p, grid, sigma):
    """Computes the effect of sigma for a state-action pair"""
    act = [i for i in range(4)]
    act.remove(action)
    reward = 0
    for a in act:
        next_state = move(pointer(p, grid.shape[0]), a)
        if grid[next_state[0]][next_state[1]] == 1 or grid[next_state[0]][next_state[1]] == 2:
            reward += -1*(sigma/4)
        elif grid[next_state[0]][next_state[1]] == 0:
            reward += -0.1*(sigma/4)
        elif grid[next_state[0]][next_state[1]] == 3:
            reward += 10*(sigma/4)
    return reward

def get_R_matrix(grid, size, actions, sigma):
    """Construct the reward matrix"""

    # Initialize reward matrix
    R = [[0 for _ in range(size)] for _ in range(len(actions))]
    for action in range(len(actions)):
        for p in range(size):
            c_from = pointer(p, grid.shape[0])
            c_to = move(pointer(p, grid.shape[0]), action)
            # Being in the target always gives no reward
            if grid[c_from[0]][c_from[1]] == 3:
                R[action][p] = 0
            # Start from an illegal cell (boundary or obstacle)
            elif (grid[c_from[0]][c_from[1]] == 1 or grid[c_from[0]][c_from[1]] == 2) and grid[c_from[0]][c_from[1]] != 3:
                R[action][p] = -1000
            # Start from a legal cell (empty) and move into an illegal cell (boundary or obstacle)
            elif (grid[c_to[0]][c_to[1]] == 1 or grid[c_to[0]][c_to[1]] == 2) and grid[c_from[0]][c_from[1]] != 3:
                R[action][p] = -1*(1-3*(sigma/4)) + remaining_reward(action, p, grid, sigma)
            # Start from a legal cell (empty) and move into a legal cell (empty)
            elif (grid[c_to[0]][c_to[1]] == 0) and grid[c_from[0]][c_from[1]] != 3:
                R[action][p] = -0.1*(1-3*(sigma/4)) + remaining_reward(action, p, grid, sigma)
            # Start from a legal cell (empty) and move into the target (dirt or charger)
            elif (grid[c_to[0]][c_to[1]] == 3 or grid[c_to[0]][c_to[1]] == 4) and grid[c_from[0]][c_from[1]] != 3:
                R[action][p] = 10*(1-3*(sigma/4)) + remaining_reward(action, p, grid, sigma)
    return R

def make_states(cols: int, rows: int):
    """Construct a state for each cell in the grid"""
    stateSpace = []
    for row in range(rows):
        for col in range(cols):
            stateSpace.append((col, row))
    return stateSpace