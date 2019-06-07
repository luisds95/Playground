'''
March 03, 2019
Luis Da Silva.

Hello world to genetic algorithms.
This file receives a target string and then uses a genetic algortihm to recreate it.
'''
import datetime as dt
import genetic


def get_fitness(target, genes):
    return sum(1 for expected, actual in zip(target, genes) if expected == actual)


def display(chromosome, startTime):
    timeDiff = dt.datetime.now() - startTime
    print("{}, {}, {}".format(''.join(chromosome.Genes), chromosome.Fitness, timeDiff))


def guess_password(target):
    geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
    start_time = dt.datetime.now()
    optimal_fitness = len(target)
    genetic.get_best(get_fitness, optimal_fitness, geneSet, display, target=target, start_time=start_time)


def test():
    target = 'Hello World!'
    guess_password(target)


if __name__ == '__main__':
    # genetic.Benchmark.run(test)
    target = 'Hello World!'
    guess_password(target)
