import numpy as np

from random import choice
from operator import attrgetter

def tournament_sel(population, tour_size=5):
    tournament = [choice(population) for _ in range(tour_size)]
    if population.optim == 'max':
        return max(tournament, key=attrgetter('fitness'))

    if population.optim == 'min':
        return min(tournament, key=attrgetter('fitness'))


# Roulette wheel
def roulette_wheel_selector(population):
    idxs = list(range(len(population)))
    indv_fitness = np.array([indv.fitness for indv in population.individuals])
    prob_of_selection = indv_fitness / indv_fitness.sum()
    selected_idx = np.random.choice(idxs, size=1, p=prob_of_selection)[0]
    return population.individuals[selected_idx]


# Elitist selection
def elitist_selector(population, pool_size=5):
    '''
    Only select best individuals of the population.
    Sort for best fitness and select randomly among them.
    '''
    elites = sorted(population, key=attrgetter('fitness'))[:pool_size]
    return choice(elites)
