import numpy as np
import random
import matplotlib.pyplot as plt
import math
import seaborn as sns
import warnings
import numpy as np
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus

warnings.filterwarnings("ignore", category=RuntimeWarning)


def vega(
    objective_functions_list,
    chromosome_type,
    lower=[],
    upper=[],
    nBits=0,
    population_size=None,
    number_of_iterations=100,
    nc=2,
    mutation_probability=0.05,
    uniform_mutation_sd=0.1,
):
    if chromosome_type == "binary":
        return binary_vega(
            objective_functions_list,
            nBits,
            population_size,
            number_of_iterations,
            mutation_probability,
        )
    elif chromosome_type == "real-valued":
        return numeric_vega(
            objective_functions_list,
            lower,
            upper,
            population_size,
            number_of_iterations,
            nc,
            uniform_mutation_sd,
        )
    else:
        raise ValueError("Unknown chromosome type")


def binary_vega(
    objective_functions_list,
    nBits,
    population_size,
    number_of_iterations,
    mutation_probability=0.05,
):
    number_of_objective_functions = len(objective_functions_list)
    subpopulation_size = int((population_size / number_of_objective_functions) * 2)
    population = [init_binary_chromosome(nBits) for _ in range(population_size)]
    statistics = {
        "min_fitness": [[] for _ in range(number_of_objective_functions)],
        "max_fitness": [[] for _ in range(number_of_objective_functions)],
        "mean_fitness": [[] for _ in range(number_of_objective_functions)],
        "sd_fitness": [[] for _ in range(number_of_objective_functions)],
    }

    for iteration in range(1, number_of_iterations + 1):
        for i in range(1, int(population_size / 2) + 1):
            parents = random.sample(range(population_size), 2)
            children = one_point_crossover(
                population[parents[0]], population[parents[1]]
            )
            population[population_size + i * 2 - 1] = children["child1"]
            population[population_size + i * 2] = children["child2"]

        population[population_size + 1 : 2 * population_size] = [
            bind_parameters(binaryMutation, probability=mutation_probability)(
                chromosome
            )
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
            selected_subpopulation = tournament_selection(subpopulation_fitness)
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

    nondominated = find_nondominated(objective_functions_values)

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


# Inicialización de cromosoma numérico (valores aleatorios dentro de los límites)
def init_numeric_chromosome(lower, upper):
    return np.random.uniform(lower, upper, size=len(lower))


# Inicialización de un cromosoma binario (valores aleatorios de 0 y 1)
def init_binary_chromosome(nBits):
    return np.random.randint(2, size=nBits)


# Operador de mutación para cromosomas binarios (cambia aleatoriamente un bit)
def binaryMutation(chromosome, probability):
    mutated_chromosome = chromosome.copy()
    for i in range(len(chromosome)):
        if random.random() < probability:
            mutated_chromosome[i] = (
                1 - mutated_chromosome[i]
            )  # Cambia de 0 a 1 o de 1 a 0 con la probabilidad dada
    return mutated_chromosome


# Función de selección para encontrar soluciones no dominadas en el espacio de Pareto
def find_nondominated(objective_functions_values):
    n = len(objective_functions_values)
    nondominated_solutions = []

    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i != j:  # No comparamos la solución con ella misma
                if np.all(
                    objective_functions_values[i] <= objective_functions_values[j]
                ) and np.any(
                    objective_functions_values[i] < objective_functions_values[j]
                ):
                    is_dominated = True
                    break  # La solución i está dominada por al menos una solución

        if not is_dominated:
            nondominated_solutions.append(i)

    return nondominated_solutions


def numeric_vega(
    objective_functions_list,
    lower,
    upper,
    population_size,
    number_of_iterations,
    nc,
    uniform_mutation_sd,
):
    if len(lower) != len(upper):
        raise ValueError("Size of lower and upper differ")

    number_of_objective_functions = len(objective_functions_list)
    subpopulation_size = int((population_size / number_of_objective_functions) * 2)
    population = [init_numeric_chromosome(lower, upper) for _ in range(population_size)]
    statistics = {
        "min_fitness": [[] for _ in range(number_of_objective_functions)],
        "max_fitness": [[] for _ in range(number_of_objective_functions)],
        "mean_fitness": [[] for _ in range(number_of_objective_functions)],
        "sd_fitness": [[] for _ in range(number_of_objective_functions)],
    }

    for _ in range(1, number_of_iterations + 1):
        for i in range(1, int(population_size / 2) + 1):
            parents = random.sample(range(population_size), 2)
            children = simulated_binary_crossover(
                population[parents[0]], population[parents[1]], nc, lower, upper
            )

            # Verifica si la longitud de population es suficiente para la asignación
            if population_size + i * 2 - 1 < len(population):
                population[population_size + i * 2 - 1] = children["child1"]
                population[population_size + i * 2] = children["child2"]
            else:
                # Si la lista es demasiado corta, agrégales a population
                population.append(children["child1"])
                population.append(children["child2"])

        population[population_size + 1 : 2 * population_size] = [
            bind_parameters(
                normally_distributed_mutation,
                sd=uniform_mutation_sd,
                lower=lower,
                upper=upper,
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
            selected_subpopulation = tournament_selection(subpopulation_fitness)
            selected_subpopulation = [
                s + (i * subpopulation_size) for s in selected_subpopulation
            ]
            selected += selected_subpopulation[: int(subpopulation_size / 2)]

        population = [population[i] for i in selected]
        objective_functions_values = objective_functions_values[selected]

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

    nondominated = find_nondominated(objective_functions_values)

    results = {
        "values": objective_functions_values[nondominated],
        "nondominated_solutions": [population[i] for i in nondominated],
        "statistics": statistics,
    }

    parameters = {
        "objective_functions_list": objective_functions_list,
        "chromosome_type": "real-valued",
        "lower": lower,
        "upper": upper,
        "population_size": population_size,
        "number_of_iterations": number_of_iterations,
        "nc": nc,
        "uniform_mutation_sd": uniform_mutation_sd,
    }

    results["parameters"] = parameters

    return results


# Función de one-point crossover para cromosomas reales
def one_point_crossover(parent1, parent2):
    n = len(parent1)
    crossover_point = random.randint(1, n - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return {"child1": child1, "child2": child2}


# Función de simulated binary crossover (SBX) para cromosomas reales
def simulated_binary_crossover(parent1, parent2, nc, lower, upper):
    u = random.random()
    if u <= 0.5:
        beta = (2.0 * u) ** (1.0 / (nc + 1))
    else:
        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (nc + 1))

    child1 = 0.5 * (((1 + beta) * parent1) + (1 - beta) * parent2)
    child2 = 0.5 * (((1 - beta) * parent1) + (1 + beta) * parent2)

    # Asegurarse de que los hijos estén dentro de los límites
    child1 = np.clip(child1, lower, upper)
    child2 = np.clip(child2, lower, upper)

    return {"child1": child1, "child2": child2}


def bind_parameters(func, **kwargs):
    def wrapped(chromosome):
        return func(chromosome, **kwargs)

    return wrapped


# Función de mutación distribuida normalmente para cromosomas reales
def normally_distributed_mutation(chromosome, sd, lower, upper):
    mutation = np.random.normal(0, sd, size=len(chromosome))
    mutated_chromosome = chromosome + mutation

    # Asegurarse de que el cromosoma mutado esté dentro de los límites
    mutated_chromosome = np.clip(mutated_chromosome, lower, upper)

    return mutated_chromosome


# Función de selección por torneo
def tournament_selection(subpopulation_fitness):
    k = 2  # Número de individuos en el torneo
    selected = []

    while len(selected) < len(subpopulation_fitness):
        competitors = random.sample(range(len(subpopulation_fitness)), k)
        winner_index = np.argmin([subpopulation_fitness[i] for i in competitors])

        # Verifica si winner_index es válido antes de acceder a competitors
        if winner_index < len(competitors):
            winner = competitors[winner_index]
            selected.append(winner)

    return selected


class ZDT1:
    # Función objetivo 1
    def func1(self, x):
        return x[0]

    # Función objetivo 2
    def func2(self, x):
        g = 1 + 9 * (sum(x[1:]) / (len(x) - 1))
        h = 1 - np.sqrt(self.func1(x) / g)
        return g * h

    def __str__(self):
        return "ZDT1"


class ZDT2:
    # Función objetivo 1
    def func1(self, x):
        return x[0]

    # Función objetivo 2
    def func2(self, x):
        g = 1 + 9 * (sum(x[1:]) / (len(x) - 1))
        h = 1 - math.pow(self.func1(x) / g, 2.0)
        return g * h

    def __str__(self):
        return "ZDT2"


class ZDT3:
    # Función objetivo 1
    def func1(self, x):
        return x[0]

    # Función objetivo 2
    def func2(self, x):
        g = 1 + 9 * (sum(x[1:]) / (len(x) - 1))
        h = (
            1
            - math.sqrt(self.func1(x) / g)
            - (self.func1(x) / g) * math.sin(10 * math.pi * self.func1(x))
        )
        return g * h

    def __str__(self):
        return "ZDT3"


class ZDT4:
    # Función objetivo 1
    def func1(self, x):
        return x[0]

    # Función objetivo 2
    def func2(self, x):
        g = (
            1.0
            + 10.0 * (len(x) - 1)
            + sum(
                [
                    math.pow(x[i], 2.0) - 10.0 * math.cos(4.0 * math.pi * x[i])
                    for i in range(1, len(x))
                ]
            )
        )
        h = 1.0 - math.sqrt(x[0] / g)
        return g * h

    def __str__(self):
        return "ZDT4"


class ZDT5:
    # Función objetivo 1
    def func1(self, x):
        return 1.0 + x[0]

    # Función objetivo 2
    def func2(self, x):
        g = sum([2 + v if v < 5 else 1 for v in x[1:]])
        h = 1 / self.func1(x)
        return g * h

    def __str__(self):
        return "ZDT5"


class ZDT6:
    # Función objetivo 1
    def func1(self, x):
        return 1 - math.exp(-4 * x[0]) * math.pow(math.sin(6 * math.pi * x[0]), 6)

    # Función objetivo 2
    def func2(self, x):
        g = 1 + 9 * math.pow(sum(x[1:]) / (len(x) - 1), 0.25)
        h = 1 - math.pow(self.func1(x) / g, 2)
        return g * h

    def __str__(self):
        return "ZDT6"


# Función para graficar las soluciones no dominadas
def plot_nondominated_solutions(x_data, y_data):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, c="b", marker="o", label="Soluciones no dominadas")
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel("Función Objetivo 1")
    plt.ylabel("Función Objetivo 2")
    plt.title("Soluciones no Dominadas en el Espacio de Pareto")
    plt.legend()
    plt.grid(True)
    plt.show()


def imprimir(ZDT, results, pf):
    # Imprime las soluciones no dominadas
    a = open(f"../results/resultados_{ZDT}.txt", "a")
    medias = {}
    medias["f1"] = 0
    medias["f2"] = 0
    a.write(f"Soluciones no dominadas encontradas: {ZDT}\n")
    a.write(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    )
    for i, _ in enumerate(results["nondominated_solutions"]):
        a.write(
            f"{i + 1}, Funcion 1: {results['values'][i][0]}, Funcion 2: {results['values'][i][1]}\n"
        )

        medias["f1"] += results["values"][i][0]
        medias["f2"] += results["values"][i][1]
    a.write(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    )
    a.write(f"Elementos: {len(results['nondominated_solutions'])}\n")
    a.write(
        f"Media Funcion 1: {medias['f1'] / len(results['nondominated_solutions'])}\n"
    )
    a.write(
        f"Media Funcion 2: {medias['f2'] / len(results['nondominated_solutions'])}\n"
    )
    a.write(f"Hipervolumen: {hypervolume(results)}\n")
    a.write(f"GDPlus: {GDPlus_Value(pf, results)}\n")
    a.write(f"IGDPlus: {IGDPlus_Value(pf, results)}\n")

    a.write("------------------------------------------------------------------\n")
    a.close()


def hypervolume(results):
    ref_point = np.array([1.2, 1.2])
    ind = HV(ref_point=ref_point)
    return ind(results["values"])


def GDPlus_Value(pf, results):
    ind = GDPlus(pf)
    return ind(results["values"])


def IGDPlus_Value(pf, results):
    ind = IGDPlus(pf)
    return ind(results["values"])


def main():
    # Define tus funciones objetivo y otros parámetros aquí
    ZDT = ZDT6()
    objective_functions_list = [
        ZDT.func1,
        ZDT.func2,
    ]  # Reemplaza con tus funciones objetivo
    chromosome_type = "real-valued"
    # array with m 0s
    m = 30
    lower = np.zeros(
        m
    )  # Reemplaza con los límites inferiores de tu espacio de búsqueda
    upper = (
        np.ones(m) * 10
    )  # Reemplaza con los límites superiores de tu espacio de búsqueda
    nBits = 0
    population_size = 300
    number_of_iterations = 100
    nc = 2
    mutation_probability = 0.05
    uniform_mutation_sd = 0.1

    # Ejecuta el algoritmo VEGA
    results = vega(
        objective_functions_list,
        chromosome_type,
        lower,
        upper,
        nBits,
        population_size,
        number_of_iterations,
        nc,
        mutation_probability,
        uniform_mutation_sd,
    )
    # Encuentra el valor mínimo y máximo en cada columna
    min_column0 = np.min(results["values"][:, 0])
    max_column0 = np.max(results["values"][:, 0])
    min_column1 = np.min(results["values"][:, 1])
    max_column1 = np.max(results["values"][:, 1])

    # Normaliza los valores en ambas columnas
    results["values"][:, 0] = (results["values"][:, 0] - min_column0) / (
        max_column0 - min_column0
    )
    results["values"][:, 1] = (results["values"][:, 1] - min_column1) / (
        max_column1 - min_column1
    )

    x_data = results["values"][:, 0]
    y_data = results["values"][:, 1]

    # Grafica las soluciones no dominadas
    # plot_nondominated_solutions(x_data, y_data)

    # The pareto front of a scaled zdt1 problem
    pf = get_problem(str(ZDT)).pareto_front()

    # plot the result
    Scatter(legend=True).add(pf, label="Pareto-front").add(
        results["values"], label="Result"
    ).show()

    imprimir(ZDT, results, pf)

    # print(x_data)
    """
    # create scatterplot with regression line
    sns.regplot(
        x=x_data,
        y=y_data,
        lowess=True,
        ci=99,
        marker="o",
        color=".3",
        line_kws=dict(color="r"),
    )
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel("Función Objetivo 1")
    plt.ylabel("Función Objetivo 2")
    plt.show()
    """


# Ejemplo de uso:
if __name__ == "__main__":
    main()
