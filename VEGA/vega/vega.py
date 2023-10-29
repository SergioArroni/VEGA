import numpy as np
from binary import binary_vega
from numeric import numeric_vega
from typing import Callable


def vega(
    objective_functions_list: list[Callable],
    chromosome_type: str,
    lower: np.ndarray,
    upper: np.ndarray,
    nBits: int,
    population_size: int,
    number_of_iterations: int,
    nc: int,
    mutation_probability: float,
    uniform_mutation_sd: float,
) -> dict:
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
