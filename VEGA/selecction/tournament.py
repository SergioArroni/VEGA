import random
import numpy as np


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
