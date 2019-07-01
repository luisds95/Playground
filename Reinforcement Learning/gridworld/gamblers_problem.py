"""
July 01, 2019.
Luis Da Silva.
luisdanield@gmail.com

This is an implementation of value iteration for finding a tabular solution.

A gambler is offered unlimited coin flips with a coin with probability p of success.
Its objective is to reach 100$.
He sets the value of his bets. If he looses then the bet amount is lost, otherwise he doubles his money.
Find an optimal solution given the capital he has.
"""

import numpy as np
import matplotlib.pyplot as plt


def expected_return(state, action, state_value, p):
    reward = 0
    next_state_value = 0

    # Evaluate positive case
    new_state_positive = state + action
    if new_state_positive >= 100:
        reward = p
    else:
        idx = int(new_state_positive - 1)
        next_state_value += state_value[idx] * p

    # Evaluate negative case
    new_state_negative = state - action
    if new_state_negative > 0:
        idx = int(new_state_negative - 1)
        next_state_value += state_value[idx] * (1-p)

    return reward + next_state_value


def gamble(p, goal, tolerance=1e-1):
    actions = np.linspace(1, goal-1, goal-1)
    policy = np.ones_like(actions, dtype=int)
    state_value = np.zeros_like(actions)

    iter = 0
    while True:
        iter += 1
        old_state_value = state_value.copy()
        old_policy = policy.copy()

        # Evaluate policy
        for state in range(1, state_value.shape[0]):
            state_value[state] = expected_return(state, policy[state], state_value, p)

        # Improve policy
        for state in range(1, state_value.shape[0]):
            rewards = []
            for action in actions:
                if action > state:
                    rewards.append(-np.inf)
                else:
                    rewards.append(expected_return(state, action, state_value, p))
            policy[state] = actions[np.argmax(rewards)]

        change = np.abs(state_value - old_state_value).max()
        policy_stable = (old_policy == policy).all()
        print(f'Iter {iter} finished! SV change: {change}. Policy stable: {policy_stable}.')
        if change <= tolerance and policy_stable:
            break

    return policy, state_value


if __name__ == '__main__':
    p = 0.45
    policy, state_value = gamble(p=p, goal=100, tolerance=1e-5)

    plt.plot(policy)
    plt.title('Policy')
    plt.xlabel('Money')
    plt.ylabel('Gamble')
    plt.savefig(f'graphs/gamblers_problem_policy_p{p}.png')
    plt.show()

    plt.plot(state_value)
    plt.axvline(p*100, 0, 1, color='black', ls='--')
    plt.title('State Value')
    plt.xlabel('Money')
    plt.ylabel('Probability of succeed')
    plt.savefig(f'graphs/gamblers_problem_state_value_p{p}.png')
    plt.show()
