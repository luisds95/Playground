# See https://deap.readthedocs.io/en/master/tutorials/basic/part2.html
import random
from deap import base, creator, tools

# First, create the individual
IND_SIZE = 5

creator.create('FitnessMinMax', base.Fitness, weights=(-1.0, 1.0))
creator.create('Individual', list, fitness=creator.FitnessMinMax)

toolbox = base.Toolbox()
toolbox.register('attr_float', random.random)
toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

# Generate a sample individual
ind1 = toolbox.individual()
print(ind1)  # list with values
print((ind1.fitness.valid))  # False because it has no value


# Fitness evaluation functions must be defined
def evaluate(ind):
    return sum(ind), len(ind)


# And the fitness must be added
ind1.fitness.values = evaluate(ind1)
print(ind1.fitness.valid)
print(ind1.fitness)

# Mutation
# Look for mutation operators in https://deap.readthedocs.io/en/master/api/tools.html#module-deap.tools
mutant = toolbox.clone(ind1)
ind2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
del mutant.fitness.values  # The fitness values are not longer correspondent to the individual
print(ind2)

# Crossover
# Look for operators in the same web as mutation
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxBlend(child1, child2, 0.5)
del child1.fitness.values
del child2.fitness.values

# Selection
# Look for operators in the same web as mutation
selected = tools.selBest([child1, child2], 2)