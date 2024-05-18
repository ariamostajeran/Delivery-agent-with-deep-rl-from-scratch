def make_states(cols: int, rows: int):
    stateSpace = []
    for row in range(rows):
        for col in range(cols):
            stateSpace.append((col, row))
    return stateSpace

def pointer(i, rows):
    return (i % (rows), i // (rows))

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
                # Have action to go down, next state is actually down and no obstacle is hit
                if action[0] == 0 and moved_down(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1
                # Have action to go up, next state is actually up and no obstacle is hit
                elif action[0] == 1 and moved_up(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1
                # Have action to go left, next state is actually left and no obstacle is hit
                elif action[0] == 2 and moved_left(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1
                # Have action to go right, next state is actually right and no obstacle is hit
                elif action[0] == 3 and moved_right(c_from, c_to) and (not hit_wall(grid, c_from, c_to)):
                    P[action[0]][p][q] = 1

    # Else we hit an obstacle, which means that our state does not change
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
    # Initialize reward matrix
    R = [[0 for _ in range(size)] for _ in range(len(actions))]
    for action in range(len(actions)):
        for p in range(size):
            c_from = pointer(p, grid.shape[0])
            c_to = move(pointer(p, grid.shape[0]), action)
            # Start from an illegal cell (boundary or obstacle)
            if grid[c_from[0]][c_from[1]] == 1 or grid[c_from[0]][c_from[1]] == 2:
                R[action][p] = -1000
            # Start from a legal cell (empty) and move into an illegal cell (boundary or obstacle)
            elif grid[c_to[0]][c_to[1]] == 1 or grid[c_to[0]][c_to[1]] == 2:
                R[action][p] = -2
            # Start from a legal cell (empty) and move into a legal cell (empty)
            elif grid[c_to[0]][c_to[1]] == 0:
                R[action][p] = -1
            # Start from a legal cell (empty) and move into the target (dirt or charger)
            elif grid[c_to[0]][c_to[1]] == 3 or grid[c_to[0]][c_to[1]] == 4:
                R[action][p] = 10
    return R
