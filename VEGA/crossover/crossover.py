import numpy as np
import random


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
