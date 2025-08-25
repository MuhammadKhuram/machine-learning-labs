# -*- coding: utf-8 -*-
"""
Created on 30-11-2024

@author: Muhammad Khuram 21013122-008
"""

import numpy as np

# Define the number of states and actions
num_states = 6  # States 0 through 5 represent the possible positions or states
num_actions = 6  # Actions leading to states 0 through 5 (each action corresponds to a transition)

# Define the rewards matrix with None for invalid actions
# The matrix represents rewards for transitioning between states. 
# None indicates an invalid transition, and numbers represent the reward for taking an action.
rewards = np.array([
    [None, None, None, None,    0, None],  # State 0: no valid actions except action 4 (reward 0)
    [None, None, None,    0, None,  100],   # State 1: valid action 3 (reward 0), action 5 (reward 100)
    [None, None, None,    0, None, None],  # State 2: valid action 3 (reward 0)
    [None,    0,    0, None,    0, None],      # State 3: valid actions 1, 2, 4 (reward 0)
    [   0, None, None,    0, None,  100],     # State 4: valid actions 0, 3 (reward 0), action 5 (reward 100)
    [None,    0, None, None,    0,  100],     # Goal State (5): valid actions 1, 4 (reward 0), action 5 (reward 100)
], dtype=object)

# Q-learning parameters
learning_rate = 0.8  # Rate at which the agent updates its knowledge
discount_factor = 0.9  # Factor by which future rewards are discounted
num_episodes = 1000  # Number of episodes to train the agent

# Initialize Q-table with zeros
Q = np.zeros((num_states, num_actions))  # Q-table holds the estimated quality of state-action pairs

# Epsilon-greedy action selection strategy
# In epsilon-greedy, with probability epsilon, a random action is selected (exploration), 
# and with probability 1 - epsilon, the best action based on the Q-table is chosen (exploitation)
def choose_action(state, epsilon):
    valid_actions = [a for a in range(num_actions) if rewards[state, a] is not None]  # Get valid actions
    if np.random.random() < epsilon:  # Exploration: randomly choose an action
        return np.random.choice(valid_actions)
    else:  # Exploitation: choose the best action based on Q-table
        return valid_actions[np.argmax([Q[state, a] for a in valid_actions])]

# Q-learning algorithm to train the agent
for episode in range(num_episodes):
    state = 0  # Start from state 0 in each episode
    epsilon = 1.0 / (episode + 1)  # Decay epsilon over time to encourage exploration early and exploitation later

    # Continue until the agent reaches the goal state (state 5)
    while state != 5:
        action = choose_action(state, epsilon)  # Choose an action using epsilon-greedy strategy
        next_state = action  # In this problem, action directly leads to the next state
        
        reward = rewards[state, action]  # Get the reward for the taken action
        # Get valid next actions from the next state
        valid_next_actions = [a for a in range(num_actions) if rewards[next_state, a] is not None]
        # Choose the best next action based on the Q-table
        best_next_action = valid_next_actions[np.argmax([Q[next_state, a] for a in valid_next_actions])]
        
        # Compute the TD-target and TD-error for Q-learning
        td_target = reward + discount_factor * Q[next_state, best_next_action]
        td_error = td_target - Q[state, action]  # The error in the Q-value estimation
        # Update the Q-value for the current state-action pair
        Q[state, action] += learning_rate * td_error

        state = next_state  # Move to the next state

# Print the learned Q-table after training
print("Learned Q-table:")
print(np.round(Q, 2))

# Determine the best path from state 0 to state 5 using the learned Q-table
state = 0  # Start from state 0
path = [state]  # Initialize the path with the starting state

# Follow the best actions from state 0 to the goal state (state 5)
while state != 5:
    valid_actions = [a for a in range(num_actions) if rewards[state, a] is not None]  # Get valid actions
    # Choose the best action based on the learned Q-values
    action = valid_actions[np.argmax([Q[state, a] for a in valid_actions])]
    state = action  # Move to the next state
    path.append(state)  # Add the next state to the path

# Print the optimal path from state 0 to state 5
print("Best path from 0 to 5:", ' -> '.join(map(str, path)))
