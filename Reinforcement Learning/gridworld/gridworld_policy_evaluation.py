'''
June 06, 2019.
Luis Da Silva.

This file evaluates a gridworld problem that is solved under a random action policy.
The outcome is a state value matrix for each of the possible states.
'''

import numpy as np
from itertools import product

WORLD_SIZE = 5
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
ACTION_PROBA = 1 / len(ACTIONS)


def is_terminal(x, y):
    return (x == 0 and y == 0) or (x == WORLD_SIZE-1 and y == WORLD_SIZE-1)


def make_action(state, action, reward=-1):
    # Returns: state, reward and whether the action was legal
    new_state = np.sum((state, action), axis=0)
    if new_state.min() < 0 or new_state.max() >= WORLD_SIZE:
        return None, 0, False
    else:
        return new_state.tolist(), reward, True


def policy_greedy(state_value_per_action):
    argmax = np.argmax(state_value_per_action)
    return state_value_per_action[argmax]


def policy_random(state_value_per_action):
    return np.sum(state_value_per_action) / len(state_value_per_action)


def iterative_state_evaluation(gamma=1, policy='greedy', verbose=10):
    if gamma < 0 or gamma > 1:
        raise ValueError('gamma must be between 0 and 1.')
    if policy.lower() == 'greedy':
        policy_func = policy_greedy
    elif policy.lower() == 'random':
        policy_func = policy_random
    elif callable(policy):
        policy_func = policy
    else:
        raise NotImplementedError('Policy "{}" has not been implemented.'.format(policy))
    print('Running "{}" policy.'.format(policy.title()))

    state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))

    iter = 0
    while True:
        old_state_values = state_values.copy()
        # For each of the possible states
        for x, y in product(range(WORLD_SIZE), range(WORLD_SIZE)):
            if is_terminal(x, y):
                # No evaluation here
                continue

            # Recalculate the state value
            state_value = []
            for action in ACTIONS:
                # Get the reward for the action and the new state
                next_state, r, legal = make_action((x, y), action)
                if legal:
                    next_x, next_y = next_state
                    # Save the value
                    state_value.append(r + gamma * state_values[next_x, next_y])

            # Action probability may be included outside the loop because this is a equiprobable problem
            state_values[x, y] = policy_func(state_value)

        # If there were no changes, stop the loop
        delta = np.sum(np.abs(state_values - old_state_values))
        iter += 1
        if delta <= 1e-3:
            print('Iter: {}. Delta: {:.2f}'.format(iter, delta))
            break
        elif verbose and (iter-1) % verbose == 0:
            print('Iter: {}. Delta: {:.2f}'.format(iter, delta))

    return state_values


if __name__ == '__main__':
    print(iterative_state_evaluation(gamma=0.9, policy='random').round(2))
