'''
March 03, 2019
Luis Da Silva.

One max problem implementation. The idea is to start from a random binary array and end
up with an array of only ones.
'''
import datetime as dt
import genetic


def get_fitness(genes, target=None):
    return sum(genes)


def display(chromosome, startTime):
    timeDiff = dt.datetime.now() - startTime
    print("{}, {}, {}".format(''.join((str(e) for e in chromosome.Genes)), chromosome.Fitness, timeDiff))


def guess_password(lenght = 10, target=None):
    geneSet = [0,1]
    start_time = dt.datetime.now()
    genetic.get_best(get_fitness, lenght, geneSet, display, start_time=start_time)


if __name__ == '__main__':
    # genetic.Benchmark.run(test)
    guess_password(100)