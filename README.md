# CougUV Markov Decision Process Simulator


## Description

Our problem is set in an underwater, GPS-denied environment. Our underwater agent, a torpedo-style AUV called CougUV, has an objective of reaching a goal position in minimum time, given a maximum threshold on ending position uncertainty. To minimize uncertainty along its path, the agent can navigate to previously known landmarks within its visibility radius. The goal of this process is to formulate this problem as a Markov Decision Process. 

Since our agent has a wide turning radius, we chose a hex grid to allow for three feasible actions from a state, given the agent’s orientation: straight ahead, straight and to the right, or straight and to the left. This allows the agent to turn a wide circle but prohibits the agent from taking sharp turns, assuming the hex cell width is equivalent to the agent’s turning radius.

Our framework for solving this problem is to use value iteration to find the shortest path. We created a reward at the goal that is 100 when our uncertainty is below our defined threshold and 0 otherwise. All other actions from a state that do not reach the goal state have a penalty of -1 to reward shorter paths.

Rather than treat viewing a landmark as a reward, we chose to include uncertainty as part of the state, which decreases when viewing a landmark. We did this to support our maximum uncertainty threshold, only rewarding the goal position when the final uncertainty is below the threshold. Adding uncertainty to the state significantly increases the size of our state space, so we implemented value function approximation to reduce computation time in finding the optimal path.


## Running the Simulator

The entire simulator is contained in a single file, `sim.py`. The top of the file has tunable parameters to adjust the simulation. The main body of the file is functions implementing the MDP, simulation, and plotting. The bottom of the file contains a main script. 


### Requirements

The simulator only requires numpy and matplotlib to be installed in a python environment:

    `pip install numpy matplotlib`


### Code Execution

Run the simulation by executing the python file: 

    `python sim.py`

For running in VSCode or other interactive IDEs, simply open the file and hit Run.


## Archived Branches

The `main` branch in the repository contains the MDP simulator with value function approximation and position uncertainty. Other branches serve as archives of the simulator at various states of development:
- The `mdp_basic` branch contains the base MDP simulation that implements value iteration. The simulator state is only the agent's pose (orientation, position). It does not have value function approximation and does not account for uncertainty or landmarks. 
- The `approx_value` branch builds on the basic MDP and implements value function approximation. It does not include landmarks or uncertainty. 
- The `state_unc` branch builds on the basic MDP to implement landmarks and add uncertainty to the state. It does not include value function approximation. 


## Bugs & To-Do

- The transition function allows for the agent to not transition states at all in some cases, leading to the agent getting stuck.
- The main branch has aesthetic improvements to plotting functions and general code structure that could be ported to other branches. 

