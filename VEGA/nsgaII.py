import platypus as plat
from random import randint, random, shuffle
from math import log, pi
from copy import copy
from terminalplot import plot
from platypus.problems import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, ZDT5
from platypus.algorithms import NSGAII
import matplotlib.pyplot as plt


def main():
    """
    SIZE_POPULATION = 30
    NUMBER_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.01

    population = initialize_population(size=SIZE_POPULATION)

    for _ in range(NUMBER_GENERATIONS):
        population = next_generation(
            population, mutation_probability=MUTATION_PROBABILITY
        )

    plot(
        [phenotype.volume for phenotype in population],
        [phenotype.surface for phenotype in population],
    )
    """
    problem = ZDT1()
    algorithm = NSGAII(problem)
    algorithm.run(10000)

    for solution in algorithm.result:
        print(solution.objectives)

    plt.scatter(
        [s.objectives[0] for s in algorithm.result],
        [s.objectives[1] for s in algorithm.result],
    )
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel("$f_1(x)$")
    plt.ylabel("$f_2(x)$")
    plt.show()


if __name__ == "__main__":
    main()
