import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

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

# MDP parameters
gamma = 0.9  # discount factor
num_value_iterations = 100  # number of iterations for value iteration
prob_veer_lr = 0.025  # probability of veering when moving forward
prob_veer_straight = 0.05  # probability of failing to turn

# Simulation parameters
start_state = (1, 0.0, 0.0)  # (direction, x, y)
goal_position = (2 * hex_width * grid_width / 2, 2 * (3 * hex_size / 2) * grid_height / 2)  # center of grid


###### Helper Functions ######
def build_states():
    '''Build all possible states in the MDP'''
    states = []
    for dir in range(6):  # 6 possible directions
        for x in range(grid_width):
            for y in range(grid_height):
                if y % 2 == 0:
                    x_offset = 0
                else:
                    x_offset = hex_width / 2
                states.append((dir, x * hex_width + x_offset, y * (3 * hex_size / 2)))
    return states

def move(state, action):
    '''Compute state after taking action'''
    dir = state[0]
    pos = state[1:]

    if action == 0:
        new_dir = dir
    elif action == 1:
        new_dir = (dir - 1) % 6
    elif action == 2:
        new_dir = (dir + 1) % 6
    else:
        raise ValueError("Invalid action")
    
    # directions: 0 to 5, starting from right and going counter-clockwise
    if new_dir == 0:
        new_pos = (pos[0] + hex_width, pos[1])
    elif new_dir == 1:
        new_pos = (pos[0] + hex_width / 2, pos[1] + 3 * hex_size / 2)
    elif new_dir == 2:
        new_pos = (pos[0] - hex_width / 2, pos[1] + 3 * hex_size / 2)
    elif new_dir == 3:
        new_pos = (pos[0] - hex_width, pos[1])
    elif new_dir == 4:
        new_pos = (pos[0] - hex_width / 2, pos[1] - 3 * hex_size / 2)
    elif new_dir == 5:
        new_pos = (pos[0] + hex_width / 2, pos[1] - 3 * hex_size / 2)

    new_state = (new_dir, new_pos[0], new_pos[1])
    return new_state

def get_neighbors(state, states):
    '''Get neighboring states reachable from current state'''
    neighbors = []
    for action in [0, 1, 2]:  # move forward, turn left, turn right
        new_state = move(state, action)
        if new_state in states:
            neighbors.append(new_state)
        # TODO: need to account for when there are no valid moves (e.g., hitting wall)
    return neighbors


###### MDP Functions ######
def transition(state, action, new_state):
    '''Probability of being in new_state given current state and action taken'''

    # if new_state is reachable from state with action, return corresponding probability
    straight = move(state, 0)
    left = move(state, 1)
    right = move(state, 2)
    rot_left = state[0] - 1 % 6
    rot_right = state[0] + 1 % 6

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
                    q += r + gamma * prob * U[s_prime] # Bellman update
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
        probs = np.array(probs)
        probs /= probs.sum()  # normalize
        idx = np.random.choice(len(next_states), p=probs)
        state = next_states[idx]
        trajectory.append(state)

    return trajectory


###### Visualization Functions ######
def plot_hex_grid():
    '''Plot the hexagonal grid'''
    plt.figure(figsize=(10, 8))
    for x in range(grid_width):
        for y in range(grid_height):
            if y % 2 == 0:
                x_offset = 0
            else:
                x_offset = hex_width / 2
            center_x = x * hex_width + x_offset
            center_y = y * (3 * hex_size / 2)
            hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=hex_size,
                                      facecolor='lightgrey', edgecolor='k')
            plt.gca().add_patch(hexagon)
    plt.xlim(-hex_width/2, grid_width * hex_width)
    plt.ylim(-hex_height/2, grid_height * (3 * hex_size / 2) - hex_size / 2)
    plt.gca().set_aspect('equal')
    plt.title('Hexagonal Grid')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(False)
    plt.show()

def plot_trajectory(trajectory):
    '''Plot the trajectory of the agent on the hex grid'''
    x_coords = [state[1] for state in trajectory]
    y_coords = [state[2] for state in trajectory]

    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, marker='o')
    plt.title('Agent Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
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
    plot_hex_grid()
    plot_trajectory(trajectory)