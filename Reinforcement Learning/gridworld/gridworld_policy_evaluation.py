'''
June 06, 2019.
Luis Da Silva.
luisdanield@gmail.com

This file evaluates a gridworld problem that is solved under a random action policy.
The outcome is a state value matrix for each of the possible states.
'''

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
ACTION_PROBA = 1 / len(ACTIONS)


def is_terminal(x, y, world_size):
    """Identify whether a state is terminal or not (top-left and bottom-right corner)"""
    return (x == 0 and y == 0) or (x == world_size-1 and y == world_size-1)


def make_action(state, action, world_size, reward=-1):
    # Returns: state, reward and whether the action was legal
    new_state = np.sum((state, action), axis=0)
    if new_state.min() < 0 or new_state.max() >= world_size:
        return None, 0, False
    else:
        return new_state.tolist(), reward, True


def policy_greedy(state_value_per_action):
    """Returns the greediest action"""
    return np.max(state_value_per_action)


def policy_random(state_value_per_action):
    """If actions are taken at random, then the value of a state is its mean"""
    return np.mean(state_value_per_action)


def iterative_state_evaluation(gamma=1, policy='greedy', world_size=5, verbose=10, max_iter=1000):
    if gamma < 0 or gamma > 1:
        raise ValueError('gamma must be between 0 and 1.')

    # Select policy
    if policy.lower() == 'greedy':
        policy_func = policy_greedy
    elif policy.lower() == 'random':
        policy_func = policy_random
    elif callable(policy):
        policy_func = policy
    else:
        raise NotImplementedError('Policy "{}" has not been implemented.'.format(policy))
    print('Running "{}" policy.'.format(policy.title()))

    state_values = np.zeros((world_size, world_size))

    iter = 0
    # Loop until the values converge or the maximum of iterations has been reached
    while iter < max_iter:
        old_state_values = state_values.copy()
        # For each of the possible states
        for x, y in product(range(world_size), range(world_size)):
            if is_terminal(x, y, world_size):
                # No evaluation here
                continue

            # Recalculate the state value
            state_value = []
            for action in ACTIONS:
                # Get the reward for the action and the new state
                next_state, r, legal = make_action((x, y), action, world_size)
                if legal:
                    next_x, next_y = next_state
                    # Save the value
                    state_value.append(r + gamma * state_values[next_x, next_y])

            # Action probability may be included outside the loop because this is a equiprobable problem
            state_values[x, y] = policy_func(state_value)

        # Calculate changes
        delta = np.sum(np.abs(state_values - old_state_values))
        iter += 1

        # If there were no changes, stop the loop
        if delta <= 1e-3:
            print('Iter: {}. Delta: {:.2f}'.format(iter, delta))
            break
        elif verbose and (iter-1) % verbose == 0:
            print('Iter: {}. Delta: {:.2f}'.format(iter, delta))

    return state_values


if __name__ == '__main__':
    state_values = iterative_state_evaluation(gamma=0.9, policy='greedy', world_size=5, verbose=1)

    sns.heatmap(state_values, cmap='bwr_r', annot=True, fmt='.2f')
    plt.title('State Values')
    plt.savefig('graphs/gridworld_greedy_policy_state_values.png')
    plt.show()
