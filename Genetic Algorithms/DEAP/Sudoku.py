'''
May 27, 2019
Luis Da Silva.

This files utilizes the DEAP framework to implement a genetic algorithm and create a Sudoku.
'''
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms


class GenerateSudoku():
    def __init__(self, sudoku_size=9, pop_size=100, cxpb=0.5, mutpb=0.2, ngen=50):
        block_size = math.sqrt(sudoku_size)
        if block_size - int(block_size) > 0:
            raise ValueError('Size must have an integer square root.')
        self.block_size = int(block_size)
        self.size = sudoku_size
        self.len = self.size ** 2
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen

        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register('attr', self.random_list)
        self.toolbox.register('individual', tools.initRepeat, creator.Individual, self.toolbox.attr, self.len)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        self.population = self.toolbox.population(n=self.pop_size)

        self.toolbox.register('mate', tools.cxOnePoint)
        self.toolbox.register('mutate', tools.mutUniformInt, low=1, up=9, indpb=self.mutpb)
        self.toolbox.register('select', tools.selTournament, tournsize=50)
        self.toolbox.register('evaluate', self.evaluate)

        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register('avg', np.mean)
        self.stats.register('std', np.std)
        self.stats.register('min', np.min)
        self.stats.register('max', np.max)

        self.hall = tools.HallOfFame(10)

        self.logbook = None

    def random_list(self):
        return random.randint(1, self.size)

    def evaluate(self, ind):
        fitness = 0
        npind = np.array(ind).reshape(self.size, self.size)

        # Evaluate rows and columns
        for i in range(self.size):
            fitness += len(set(npind[i, :])) - self.size + 1
            fitness += len(set(npind[:, i])) - self.size + 1

        # Evaluate block
        for mblock in npind.reshape(self.block_size, self.block_size, self.block_size, self.block_size):
            blocks = [[] for _ in range(self.block_size)]
            for i, b in enumerate(mblock):
                blocks[i].extend(b)
            for block in mblock:
                fitness += len(set(block.reshape(self.size))) - self.size + 1

        return fitness,

    def run(self):
        _, self.logbook = algorithms.eaSimple(self.population, self.toolbox, cxpb=self.cxpb,
                                           mutpb=self.mutpb, ngen=self.ngen, stats=self.stats,
                                           halloffame=self.hall, verbose=True)


def main(sudoku_size=9, pop_size=100, cxpb=0.25, mutpb=0.1, ngen=50):
    sudoku = GenerateSudoku(sudoku_size=sudoku_size, pop_size=pop_size, cxpb=cxpb, mutpb=mutpb, ngen=ngen)
    sudoku.run()
    print(np.array(sudoku.hall[0]).reshape(9,9))

    avg, max = sudoku.logbook.select("avg", "max")
    plt.plot(avg, label='Average')
    plt.plot(max, label='Max')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()


if __name__ == '__main__':
    main(pop_size=100, ngen=10000)