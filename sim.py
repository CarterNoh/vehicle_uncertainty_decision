import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.cm as cm
import matplotlib.colors as mcolors

'''
##### CougUV Markov Decision Process Simulator #####
World: hexagonal grid with landmark(s) and goal
State: direction agent is facing, cell coordinates of the agent
Action: move forward, turn left, turn right
Transition model: mostly deterministic, chance of veering left or right when moving forward, or going straight when turning

Goal: Implement value iteration to find optimal policy for reaching goal from any starting state
'''

# TODO: Does not account for the robot hitting the wall, when no moves are possible. We need to add some kind of "backup actions" 
# where the vehicle just turns left or right. Or could encode that into our neighbor function and account for it somehow in the transition probabilities.


###### Parameters ######
# Grid parameters (pointy-top hex grid)
grid_width = 5  # width of hex grid
grid_height = 5  # height of hex grid
hex_size = 1.0  # size of one side of hex cell
hex_height = 2 * hex_size
hex_width = np.sqrt(3) * hex_size
EVEN_ROW_DELTAS = [
    (1, 0), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1)
] # neighbor deltas for even rows because of grid system

ODD_ROW_DELTAS = [
    (1, 0), (1, 1), (0, 1),
    (-1, 0), (0, -1), (1, -1)
] # neighbor deltas for odd rows because of grid system 

# MDP parameters
gamma = 0.9  # discount factor
num_value_iterations = 100  # number of iterations for value iteration
prob_veer_lr = 0.025  # probability of veering when moving forward
prob_veer_straight = 0.05  # probability of failing to turn

# Simulation parameters
start_state = (1, 0, 0)  # (direction robot is facing, gx, gy)
goal_position = (4, 4)  # (gx, gy) position of goal


###### Helper Functions ######
def in_bounds(gx, gy):
    '''Check if position is within grid bounds'''
    if 0 <= gx < grid_width and 0 <= gy < grid_height:
        return True
    else:
        return False


def build_states():
    '''Build all possible states in the MDP'''
    states = []
    for dir in range(6):  # 6 possible directions
        for gx in range(grid_width):
            for gy in range(grid_height):
                states.append((dir, gx, gy))
    return states

def move(state, action):
    '''Compute state after taking action'''
    dir, gx, gy = state

    if action == 0: # move forward
        new_dir = dir
    elif action == 1: # turn left
        new_dir = (dir - 1) % 6
    elif action == 2: # turn right
        new_dir = (dir + 1) % 6
    else:
        raise ValueError("Invalid action")
    
    deltas = EVEN_ROW_DELTAS if gy % 2 == 0 else ODD_ROW_DELTAS
    dx, dy = deltas[new_dir]

    new_gx = gx + dx
    new_gy = gy + dy

    # wall check
    if not in_bounds(new_gx, new_gy):
        # hit wall: stay in place, but orientation updates
        return (new_dir, gx, gy)

    new_state = (new_dir, new_gx, new_gy)
    return new_state

def get_neighbors(state, states):
    '''Get neighboring states reachable from current state'''
    neighbors = set()
    neighbors.add(state)  # include current state (hit a wall)

    for action in [0, 1, 2]:  # move forward, turn left, turn right
        new_state = move(state, action)
        neighbors.add(new_state)
    return list(neighbors)


###### MDP Functions ######
def transition(state, action, new_state):
    '''Probability of being in new_state given current state and action taken'''

    # if new_state is reachable from state with action, return corresponding probability
    straight = move(state, 0)
    left = move(state, 1)
    right = move(state, 2)
    rot_left = (state[0] - 1) % 6
    rot_right = (state[0] + 1) % 6

    if straight == left == right == state: # hit wall, stay in place
        return 1.0 if new_state == state else 0.0

    if action == 0:  # move forward
        if new_state == straight:
            return 1 - 2 * prob_veer_lr
        elif new_state == left or new_state == right:
            return prob_veer_lr
        else:
            return 0.0
    elif action == 1:  # turn left
        if new_state == left:
            return 1 - prob_veer_straight
        elif new_state == straight:
            return prob_veer_straight
        else: 
            return 0.0
    elif action == 2:  # turn right
        if new_state == right:
            return 1 - prob_veer_straight
        elif new_state == straight:
            return prob_veer_straight
        elif new_state[0] == rot_right and new_state[1:] == state[1:]:
            return 0.0
        else:
            return 0.0
    elif new_state == state:
        return 0.0

def reward(state, action, new_state):
    if (new_state[1], new_state[2]) == goal_position:
        return 100  # reward for reaching goal
    else:
        return -1  # small penalty for each step to encourage faster reaching of goal

def value_iteration(states, actions, transition, reward, gamma, tolerance=1e-6):
    U = {s: 0 for s in states}  # initialize value function
    policy = {s: 0 for s in states}  # initialize policy

    count = 0
    while count < num_value_iterations:
        delta = 0
        for s in states:
            u = U[s] # store old value
            Q = []
            neighbors = get_neighbors(s, states)
            for a in actions:
                q = 0
                for s_prime in neighbors:
                    prob = transition(s, a, s_prime)
                    r = reward(s, a, s_prime)
                    # q += r + gamma * prob * U[s_prime] # Bellman update
                    q += prob * (r + gamma * U[s_prime])
                Q.append(q)
            U[s] = max(Q)
            policy[s] = np.argmax(Q)
            delta = max(delta, abs(u - U[s])) # convergence check
        count += 1
        if delta < tolerance:
            break

    return policy, U

def simulate(states, start_state, policy, steps):
    '''Simulate agent following policy from start_state for given number of steps'''
    state = start_state
    trajectory = [state]

    for _ in range(steps):
        action = policy[state]
        # sample new state based on transition probabilities
        probs = []
        next_states = []
        for s_prime in get_neighbors(state, states):
            prob = transition(state, action, s_prime)
            if prob > 0:
                probs.append(prob)
                next_states.append(s_prime)

        if not next_states: # Safe-guard: no valid moves
            state = state  #stay in place
            trajectory.append(state)
            continue

        probs = np.array(probs)
        probs /= probs.sum()  # normalize
        idx = np.random.choice(len(next_states), p=probs)
        state = next_states[idx]
        trajectory.append(state)
        if state[1:] == goal_position:
            break  # stop if goal is reached

    return trajectory


###### Visualization Functions ######
def grid_to_xy(gx, gy):
    '''Convert grid coordinates to x, y positions for plotting'''
    x_offset = 0 if gy % 2 == 0 else hex_width / 2
    x = gx * hex_width + x_offset
    y = gy * (3 * hex_size / 2)
    return x, y

def collapse_U_to_grid(U, grid_width, grid_height, directions):
    '''Collapse utility values U over directions to get grid utility for plotting'''
    V = np.full((grid_width, grid_height), -np.inf)

    for (d, gx, gy), val in U.items():
        V[gx, gy] = max(V[gx, gy], val)

    return V

def plot_hex_grid(U, goal_position=None, trajectory=None):
    '''Plot the hexagonal grid'''
    # Normalize utility values for coloring
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = mcolors.Normalize(vmin=min(U.flatten()), vmax=max(U.flatten()))
    cmap = cm.get_cmap('viridis')
    
    for x in range(grid_width):
        for y in range(grid_height):
            if y % 2 == 0:
                x_offset = 0
            else:
                x_offset = hex_width / 2
            center_x = x * hex_width + x_offset
            center_y = y * (3 * hex_size / 2)
            value = U[x, y]
            color = cmap(norm(value))
            hexagon = RegularPolygon(
                (center_x, center_y), 
                numVertices=6, 
                radius=hex_size,
                facecolor=color, 
                edgecolor='k',
                linewidth=1
            )

            # Highlight goal position
            if goal_position is not None and (x, y) == goal_position:
                hexagon.set_edgecolor('red')
                hexagon.set_linewidth(3)
                # hexagon.set_facecolor('gold')
            ax.add_patch(hexagon)
    
    # Plot trajectory if provided
    if trajectory is not None:
        xs, ys = zip(*[grid_to_xy(gx, gy) for (_, gx, gy) in trajectory])
        ax.plot(xs, ys, '-o', color='black', linewidth=2, markersize=6)

    #Add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for older matplotlib versions

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Value Function V')

    # Finish plotting
    ax.set_xlim(-hex_width/2, grid_width * hex_width)
    ax.set_ylim(-hex_height/2, grid_height * (3 * hex_size / 2) - hex_size / 2)
    ax.set_aspect('equal')
    ax.set_title('Hexagonal Grid')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(False)

    plt.show()


###### Main Execution ######
if __name__ == "__main__":
    states = build_states()
    actions = [0, 1, 2]  # 0: move forward, 1: turn left, 2: turn right

    # Compute optimal policy using value iteration
    policy, U = value_iteration(states, actions, transition, reward, gamma)

    # # Simulate agent following optimal policy
    trajectory = simulate(states, start_state, policy, steps=50)

    # Plot results
    V = collapse_U_to_grid(U, grid_width, grid_height, directions=6)
    plot_hex_grid(V, goal_position, trajectory)