import numpy as np
import random
import mutation, selecction, util, crossover


def binary_vega(
    objective_functions_list,
    nBits,
    population_size,
    number_of_iterations,
    mutation_probability=0.05,
):
    number_of_objective_functions = len(objective_functions_list)
    subpopulation_size = int((population_size / number_of_objective_functions))
    population = [init_binary_chromosome(nBits) for _ in range(population_size)]
    statistics = {
        "min_fitness": [[] for _ in range(number_of_objective_functions)],
        "max_fitness": [[] for _ in range(number_of_objective_functions)],
        "mean_fitness": [[] for _ in range(number_of_objective_functions)],
        "sd_fitness": [[] for _ in range(number_of_objective_functions)],
    }

    for _ in range(1, number_of_iterations + 1):
        population_size = len(population)
        for i in range(1, int(population_size / 2) + 1):
            parents = random.sample(range(population_size), 2)
            children = crossover.one_point_crossover(
                population[parents[0]], population[parents[1]]
            )
            population[population_size + i * 2 - 1] = children["child1"]
            population[population_size + i * 2] = children["child2"]

        population[population_size + 1 : 2 * population_size] = [
            util.bind_parameters(
                mutation.binaryMutation, probability=mutation_probability
            )(chromosome)
            for chromosome in population[population_size + 1 : 2 * population_size]
        ]

        random.shuffle(population)

        objective_functions_values = np.array(
            [
                list(map(lambda f: f(chromosome), objective_functions_list))
                for chromosome in population
            ]
        )

        selected = []

        for i in range(number_of_objective_functions):
            subpopulation_fitness = objective_functions_values[
                (i * subpopulation_size) : ((i + 1) * subpopulation_size), i
            ]
            selected_subpopulation = selecction.tournament_selection(
                subpopulation_fitness
            )
            selected_subpopulation = [
                s + (i * subpopulation_size) for s in selected_subpopulation
            ]
            selected += selected_subpopulation[: int(subpopulation_size / 2)]

        # population = [population[i] for i in selected]
        # objective_functions_values = objective_functions_values[selected]

        for i in range(number_of_objective_functions):
            statistics["min_fitness"][i].append(
                np.min(objective_functions_values[:, i])
            )
            statistics["max_fitness"][i].append(
                np.max(objective_functions_values[:, i])
            )
            statistics["mean_fitness"][i].append(
                np.mean(objective_functions_values[:, i])
            )
            statistics["sd_fitness"][i].append(np.std(objective_functions_values[:, i]))

    nondominated = util.find_nondominated(objective_functions_values)

    results = {
        "values": objective_functions_values[nondominated],
        "nondominated_solutions": [population[i] for i in nondominated],
        "statistics": statistics,
    }

    parameters = {
        "objective_functions_list": objective_functions_list,
        "chromosome_type": "binary",
        "nBits": nBits,
        "population_size": population_size,
        "number_of_iterations": number_of_iterations,
        "mutation_probability": mutation_probability,
    }

    results["parameters"] = parameters

    return results


# Inicialización de un cromosoma binario (valores aleatorios de 0 y 1)
def init_binary_chromosome(nBits):
    return np.random.randint(2, size=nBits)
