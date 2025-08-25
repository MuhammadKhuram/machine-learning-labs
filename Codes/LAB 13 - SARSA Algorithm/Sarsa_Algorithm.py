# -*- coding: utf-8 -*-
"""
Created on 30-11-2024

@author: Muhammad Khuram 21013122-008
"""

import numpy as np

def SARSA():
    # Reward matrix: Defines the rewards for state-action pairs
    reward_matrix = np.array([
        [-5, 0, -np.inf, -np.inf, -np.inf, -np.inf],  # Rewards for state 0
        [0, -5, 0, 0, -np.inf, -np.inf],               # Rewards for state 1
        [-np.inf, 0, -5, 0, -np.inf, 100],             # Rewards for state 2
        [-np.inf, 0, 0, -5, 0, -np.inf],               # Rewards for state 3
        [-np.inf, -np.inf, -np.inf, 0, -5, 100],       # Rewards for state 4
        [-np.inf, -np.inf, 0, -np.inf, -np.inf, 0]     # Rewards for state 5 (terminal state)
    ])
    
    # Transition matrix: Defines valid transitions between states
    transition_matrix = np.array([
        [1, 1, 0, 0, 0, 0],  # Transitions from state 0
        [1, 1, 1, 1, 0, 0],  # Transitions from state 1
        [0, 1, 1, 1, 0, 1],  # Transitions from state 2
        [0, 1, 1, 1, 1, 0],  # Transitions from state 3
        [0, 0, 0, 1, 1, 1],  # Transitions from state 4
        [0, 0, 1, 0, 1, 1]   # Transitions from state 5 (terminal state)
    ])
    
    # Number of states and actions
    n_states = reward_matrix.shape[0]
    n_actions = reward_matrix.shape[1]
    
    # Initialize Q-table with small random values
    Q = np.random.uniform(-0.05, 0.05, size=(n_states, n_actions))
    
    # Hyperparameters
    learning_rate = 0.7  # Mu: Step size for Q-value updates
    discount_factor = 0.4  # Gamma: Importance of future rewards
    epsilon = 0.1  # Epsilon: Exploration rate for epsilon-greedy policy
    max_iterations = 1000  # Number of training episodes
    
    # SARSA Algorithm
    for iteration in range(max_iterations):
        # Initialize the state randomly
        state = np.random.randint(n_states)
        
        # Choose an action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            # Explore: Randomly select a valid action
            valid_actions = np.where(transition_matrix[state, :] != 0)[0]
            action = np.random.choice(valid_actions)
        else:
            # Exploit: Choose the action with the highest Q-value
            action = np.argmax(Q[state, :])
        
        # Continue until reaching the terminal state (state 5)
        while state != 5:  # State 5 is the terminal state
            # Get the reward and determine the next state based on the action
            reward = reward_matrix[state, action]
            next_state = action  # In this example, the next state is the chosen action
            
            # Choose the next action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                # Explore: Randomly select a valid action for the next state
                valid_actions = np.where(transition_matrix[next_state, :] != 0)[0]
                next_action = np.random.choice(valid_actions)
            else:
                # Exploit: Choose the action with the highest Q-value for the next state
                next_action = np.argmax(Q[next_state, :])
            
            # Update Q-value using the SARSA update rule
            Q[state, action] += learning_rate * (
                reward + discount_factor * Q[next_state, next_action] - Q[state, action]
            )
            
            # Move to the next state and action
            state = next_state
            action = next_action
    
    # Print the resulting Q-table after training
    print("Q-table after training:")
    print(np.round(Q, 1))

# Call the SARSA function to execute
SARSA()
