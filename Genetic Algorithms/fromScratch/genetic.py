'''
March 03, 2019
Luis Da Silva

Parent file with classes for implementation of genetic algorithms from scratch
'''
import random
import datetime as dt
import time
import numpy as np
import sys


class Chromosome:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness


class Benchmark:
    @staticmethod
    def run(function):
        timings = []
        stdout = sys.stdout
        for i in range(100):
            sys.stdout = None
            start_time = time.time()
            function()
            seconds = time.time() - start_time
            timings.append(seconds)
            mean = np.mean(timings)
            sys.stdout = stdout
            print('{} {:3.2f} {:3.2f}'.format(1 + i, mean, np.std(timings) if i>1 else 0))


def _generate_parent(length, geneSet, target, get_fitness):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    fitness = get_fitness(genes, target=target)
    return Chromosome(genes, fitness)


def _mutate(genes, geneSet, target, get_fitness):
    index = random.randrange(0, len(genes))
    newGene, alternate = random.sample(geneSet, 2)
    genes[index] = alternate if newGene == genes[index] else newGene
    fitness = get_fitness(target=target, genes=genes)
    return Chromosome(genes, fitness)


def get_best(get_fitness, optimalFitness, geneSet, display, target=None, start_time=None):
    random.seed()
    if start_time is None:
        start_time = dt.datetime.now()
    if isinstance(geneSet, str):
        geneSet = list(geneSet)
    bestParent = _generate_parent(optimalFitness, geneSet, target, get_fitness)
    display(bestParent, start_time)

    while True:
        child = _mutate(bestParent.Genes, geneSet, target, get_fitness)
        if bestParent.Fitness >= child.Fitness:
            continue

        display(child, start_time)
        if child.Fitness >= optimalFitness:
            break
        bestParent = child