import numpy as np
import random


# Función de mutación distribuida normalmente para cromosomas reales
def normally_distributed_mutation(
    chromosome: list, sd: float, lower: np.ndarray, upper: np.ndarray
) -> list:
    mutation = np.random.normal(0, sd, size=len(chromosome))
    mutated_chromosome = chromosome + mutation

    # Asegurarse de que el cromosoma mutado esté dentro de los límites
    mutated_chromosome = np.clip(mutated_chromosome, lower, upper)

    return mutated_chromosome


# Operador de mutación para cromosomas binarios (cambia aleatoriamente un bit)
def binaryMutation(chromosome: list, probability: float) -> list:
    mutated_chromosome = chromosome.copy()
    for i in range(len(chromosome)):
        if random.random() < probability:
            mutated_chromosome[i] = (
                1 - mutated_chromosome[i]
            )  # Cambia de 0 a 1 o de 1 a 0 con la probabilidad dada
    return mutated_chromosome
