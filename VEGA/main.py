import test, vega, util
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    # Define tus funciones objetivo y otros parámetros aquí
    ZDT = test.ZDT1()
    objective_functions_list = [
        ZDT.func1,
        ZDT.func2,
    ]  # Reemplaza con tus funciones objetivo
    chromosome_type = "real-valued"

    lower = np.zeros(
        ZDT.m
    )  # Reemplaza con los límites inferiores de tu espacio de búsqueda
    upper = (
        np.ones(ZDT.m) * 1
    )  # Reemplaza con los límites superiores de tu espacio de búsqueda
    nBits = 0
    population_size = 100
    number_of_iterations = 250
    nc = 2
    mutation_probability = 0.01
    uniform_mutation_sd = 0.01

    # Ejecuta el algoritmo VEGA
    results = vega.vega(
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

    # util.normalizar(results)

    # The pareto front of a scaled zdt1 problem
    pf = util.get_pf(ZDT)

    util.plot_Scatter(pf, results, ZDT)

    util.imprimir(ZDT, results, pf)


# Ejemplo de uso:
if __name__ == "__main__":
    main()
