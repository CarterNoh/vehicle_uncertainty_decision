import numpy as np
import matplotlib.pyplot as plt
import hexalattice.hexalattice as hl


##### CougUV Markov Decision Process Simulator #####

# World: hexagonal grid (flat-topped) with landmark(s) and goal (using hexalattice package: https://pypi.org/project/hexalattice/)
# State: direction agent is facing and cell position of the agent
# Action: move forward, turn left, turn right
# Transition model: mostly deterministic - chance of veering left or right when moving forward, or going straight when turning




gamma = 0.9  # discount factor
prob_move_fail = 0.025  # probability of veering when moving forward
prob_turn_fail = 0.05  # probability of failing to turns


def transition(state, action):
    dir = state[0]
    pos = state[1]
    if action == 0:  # move forward
        if np.random.rand() < prob_move_fail:
            pass
        elif np.random.rand() < (1-prob_move_fail):
            pass
        else:
            pass
    elif action == 1:  # turn left
        if np.random.rand() < prob_turn_fail:
            pass
        else:
            pass
    elif action == 2:  # turn right
        if np.random.rand() < prob_turn_fail:
            pass
        else:
            pass
