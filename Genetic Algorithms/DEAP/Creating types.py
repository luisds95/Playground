# See https://deap.readthedocs.io/en/master/tutorials/basic/part1.html
from deap import base, creator, tools
import random

# Creates a fitness measure called "FitnessMin" which needs to be minimized
# [weights=(-1.0,)]. This is a tuple to include the multi-objective case.
# Weights can also be used to vary the importance of the objective.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))

# Individuals are created from the list class
creator.create('Individual', list, fitness=creator.FitnessMax)

# The individual will be a list of 10 float numbers
IND_SIZE = 10

toolbox = base.Toolbox()
toolbox.register('attr_float', random.random)
toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

# Population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
population = toolbox.population(n=100)