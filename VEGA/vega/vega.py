from binary import binary_vega
from numeric import numeric_vega

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
