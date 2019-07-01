"""
June 29, 2019.
Luis Da Silva.
luisdanield@gmail.com

This is an implementation of policy iteration for finding a tabular solution.

As described in Sutton's book, page 81. Jack has a car rental with two locations,
he receives an amount d for every rented car. If there are no cars available then he
receives nothing. He may have up to n cars in stock between both locations. m cars may
be moved between locations each night. Rentals and requests follow a poisson distribution.
The idea is to maximize jack's profit.
"""
import numpy as np
from scipy.stats import poisson
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

RENTAL_PRICE = 10
MOVE_COST = 2
STOCK_LIMIT = 20
MOVE_LIMIT = 5
REQUEST_LAMBDAS = (3, 4)
RETURN_LAMBDAS = (3, 2)
DISCOUNT_RATE = 0.9
TOLERANCE = 1e-1
POISSON_TOLERANCE = 1e-4
COST_STORAGE = 4


def get_poisson_dict(l, ns):
    """
    Calculate probabilities from a Poisson distribution.
    :param l: int. Expected number.
    :param ns: int. Max consecutive possibility
    :return: dictionary with {number: probability} for each n from 0 to ns.
    """
    total = 0
    d = {}

    # Calculate poisson probabilities
    for n in range(ns+1):
        proba = poisson.pmf(n, l)

        # Speed up computation by smoothing small values
        if proba >= POISSON_TOLERANCE:
            total += proba
            d[n] = proba

    # Normalise probabilities
    for k, v in d.items():
        d[k] = v/total

    return d


def expected_reward(state, action, state_value, request_probas, return_probas, n_free, n_max_storage):
    """
    Use to evaluate a single action in a single state.
    :param state: tuple with (number of first location's cars, number of second location's cars)
    :param action: int representing the number of cars to move from location 1 to 2.
    :param state_value: state value matrix.
    :param request_probas: dict with probabilities of request.
    :param return_probas: dict with probabilities of return.
    :param n_free: number of cars that one may move from one location to the other free of charge.
    :param n_max_storage: number of cars allowed to expend the night free of charge.
    :return: expected reward for the action in the given state.
    """
    # Execute the action
    first_cars = min(state[0] - action, STOCK_LIMIT)
    second_cars = min(state[1] + action, STOCK_LIMIT)
    moved_cars = abs(state[0] - first_cars)
    paid_moved_cars = max(moved_cars - n_free, 0)
    reward = paid_moved_cars * -MOVE_COST

    # Loop to get the value of each possibilities for rental
    for rq1, rq1_proba in request_probas[0].items():
        for rq2, rq2_proba in request_probas[1].items():
            # Rentals in location one and two are independent
            request_proba = rq1_proba * rq2_proba

            # Rent reward
            rented1 = min(rq1, first_cars)
            rented2 = min(rq2, second_cars)
            new_first_cars = first_cars - rented1
            new_second_cars = second_cars - rented2
            rent_reward = (rented1 + rented2) * RENTAL_PRICE

            # Loop to get the value of each possible return
            for rt1, rt1_proba in return_probas[0].items():
                for rt2, rt2_proba in return_probas[1].items():
                    # Returns in location one and two are independent
                    return_proba = rt1_proba * rt2_proba

                    # Update number of cars
                    new_first_cars = min(new_first_cars + rt1, STOCK_LIMIT)
                    new_second_cars = min(new_second_cars + rt2, STOCK_LIMIT)

                    # Update storage cost
                    storage_cost = 0
                    if new_first_cars > n_max_storage:
                        storage_cost += 4
                    if new_second_cars > n_max_storage:
                        storage_cost += 4

                    # Update state reward
                    reward += (request_proba * return_proba) * \
                              (rent_reward + storage_cost +
                               DISCOUNT_RATE * state_value[new_first_cars, new_second_cars])

    return reward


def optimise_rental(n_free=0, n_max_storage=None):
    """
    Optimise the rental problem.
    :param n_free: number of cars that one may move from one location to the other free of charge.
    :param n_max_storage: number of cars allowed to expend the night free of charge.
    :return: policy and state value matrices.
    """
    state_value = np.zeros((STOCK_LIMIT+1, STOCK_LIMIT+1))
    policy = np.zeros((STOCK_LIMIT+1, STOCK_LIMIT+1), dtype=np.int)
    actions = [(-MOVE_LIMIT + i) for i in range(MOVE_LIMIT*2 + 1)]

    # Calculate the poisson probabilities of each outcome for the store 0 and 1 and normalise their probabilities
    request_probas = []
    for l in REQUEST_LAMBDAS:
        request_probas.append(get_poisson_dict(l, STOCK_LIMIT))
    return_probas = []
    for l in RETURN_LAMBDAS:
        return_probas.append(get_poisson_dict(l, STOCK_LIMIT))

    # Evaluate subsequently improved policies until no improvements can be done
    improvement = True
    iter = 0
    while improvement:
        iter += 1
        improvement = False
        state_change = TOLERANCE * 2
        print('*--'*10)
        print('Begining policy evaluation...')

        # Evaluate the current policy
        eval_iters = 0
        while state_change > TOLERANCE:
            old_state_value = state_value.copy()
            for n_first, n_second in product(range(STOCK_LIMIT+1), range(STOCK_LIMIT+1)):
                state_value[n_first, n_second] = expected_reward((n_first, n_second), policy[n_first, n_second],
                                                                 state_value, request_probas, return_probas,
                                                                 n_free, n_max_storage)
            state_change = abs(old_state_value - state_value).max()
            eval_iters += 1
            print(state_change)
        print('Policy evaluation finished!')

        # Use new values to improve the policy
        with tqdm(total=(STOCK_LIMIT+1)**2, desc="Improving policy") as pbar:
            for n_first, n_second in product(range(STOCK_LIMIT + 1), range(STOCK_LIMIT + 1)):
                actions_reward = []
                for action in actions:
                    # If we have enough cars to perform the action
                    if (0 <= action <= n_first) or (-n_second <= action <= 0):
                        actions_reward.append(expected_reward((n_first, n_second), action, state_value,
                                                              request_probas, return_probas, n_free, n_max_storage))
                    else:
                        actions_reward.append(-np.inf)
                pbar.update(1)

                # Select best action
                best_action = actions[np.argmax(actions_reward)]
                if policy[n_first, n_second] != best_action:
                    policy[n_first, n_second] = best_action
                    improvement = True

        print(f'\nIteration {iter} finished! Policy improved: {improvement}. Evaluation iters: {eval_iters}')

    return policy, state_value


if __name__ == '__main__':
    n_free = 1
    n_max_storage = 10
    policy, state_value = optimise_rental(n_free, n_max_storage)

    # Plot results
    ax = sns.heatmap(policy, cmap='bwr', center=0, annot=True)
    ax.invert_yaxis()
    plt.title('Policy')
    plt.ylabel('Cars in first location')
    plt.xlabel('Cars in second location')
    plt.savefig(f'graphs/jacks_move{MOVE_LIMIT}_limit{STOCK_LIMIT}_{n_free}_{n_max_storage}_policy.png')
    plt.show()

    ax = sns.heatmap(state_value, cmap='Blues')
    ax.invert_yaxis()
    plt.title('State Values')
    plt.ylabel('Cars in first location')
    plt.xlabel('Cars in second location')
    plt.savefig(f'graphs/jacks_move{MOVE_LIMIT}_limit{STOCK_LIMIT}_{n_free}_{n_max_storage}_state_values.png')
    plt.show()
