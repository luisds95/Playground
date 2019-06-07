'''
March 14, 2019
Luis Da Silva.

Genetic implementation of a strategy which solves the Prisoners dilemma.
See: https://en.wikipedia.org/wiki/Prisoner%27s_dilemma
'''
from itertools import product, combinations
import random
import numpy as np
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.score = 0

    def record_game(self, a1, a2):
        self.genes = self.genes[2:7] + a1 + a2 + self.genes[7:]

    def get_action(self, combis):
        idx = combis.index(tuple(self.genes[:6])) + 6
        return self.genes[idx]

    def record_score(self, score):
        self.score += score

    def finish_round(self):
        self.fitness = self.score
        self.score = 0


def play_round(a1, a2):
    if a1 == 'C':
        if a1 == a2:
            return 4, 4
        else:
            return 0, 5
    elif a1 == a2:
        return 1, 1
    else:
        return 5, 0


def play_against(p1, p2, n, combis):
    for i in range(n):
        a1 = p1.get_action(combis)
        a2 = p2.get_action(combis)
        s1, s2 = play_round(a1, a2)

        p1.record_score(s1)
        p2.record_score(s2)

        p1.record_game(a1, a2)
        p2.record_game(a2, a1)


def get_fitness(generation, combis, n=100):
    all_fitness = []
    matchs = combinations(generation, 2)
    for players in matchs:
        play_against(players[0], players[1], n, combis)
    for player in generation:
        player.finish_round()
        all_fitness.append(player.fitness)
    return all_fitness


def mutate(ind, p):
    for i in range(len(ind.genes)):
        if random.random() > p:
            continue
        if ind.genes[i] == 'C':
            new_genes = list(ind.genes)
            new_genes[i] = 'D'
            ind.genes = ''.join(new_genes)
        else:
            new_genes = list(ind.genes)
            new_genes[i] = 'C'
            ind.genes = ''.join(new_genes)


def crossover(ind1, ind2, p):
    if random.random() > p:
        return Individual(ind1.genes)
    else:
        split = random.randint(0, len(ind1.genes))
        genes = ind1.genes[:split]
        genes += ind2.genes[split:]
        return Individual(genes)


def gen_parent(geneset='DC', n=70):
    return Individual(''.join([random.choice(geneset) for _ in range(n)]))


def gen_generation(n):
    return [gen_parent() for _ in range(n)]


def new_generation(generation, all_fitness, pc, pm):
    new_gen = [generation[int(np.argmax(all_fitness))]]
    total = all_fitness.sum()
    cumsum = all_fitness.cumsum()

    for i in range(len(generation)-1):
        r1 = random.randint(0, total)
        r2 = random.randint(0, total)
        ind1 = next(generation[idx] for idx, value in enumerate(cumsum) if value >= r1)
        ind2 = next(generation[idx] for idx, value in enumerate(cumsum) if value >= r2)
        new_ind = crossover(ind1, ind2, pc)
        mutate(new_ind, pm)
        new_gen.append(new_ind)

    return new_gen


def run(n_generations = 100, n_individuals = 20, pc=0.7, pm=0.001, verbose=1):
    generation = gen_generation(n_individuals)
    combis = [i for i in product('CD', repeat=6)]
    avg_fitness = []
    max_fitness = []
    all_max = 0
    for i in range(n_generations):
        all_fitness = np.array(get_fitness(generation, combis))
        generation = new_generation(generation, all_fitness, pc, pm)

        avg_fitness.append(all_fitness.mean())
        gen_max = all_fitness.max()
        max_fitness.append(gen_max)
        if gen_max>all_max:
            all_max = gen_max
            best_str = generation[int(np.argmax(all_fitness))].genes

        if i%verbose == 0:
            print('Generation {} results'.format(i+1))
            print('Max fitness: {}'.format(gen_max))
            print('Average fitness: {}'.format(all_fitness.mean()))
            print('Best Strategy: {}'.format(generation[int(np.argmax(all_fitness))].genes))
            print('*--'*30)

    print('Best overall fitness: {}'.format(np.max(max_fitness)))
    print('Best overall strategy: {}'.format(best_str))

    plt.plot(avg_fitness, label='Average')
    plt.plot(max_fitness, label='Max')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run(100, pc=0.7, pm=.01, verbose=50)