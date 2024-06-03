import numpy as np

from random import choice
from operator import attrgetter

def tournament_sel(population, tour_size=5):
    '''
    Same implementation from the lab charles module.
    n individuals are randomly chosen from the population pool and the best one
        among them is always picked.
    '''
    tournament = [choice(population) for _ in range(tour_size)]
    if population.optim == 'max':
        return max(tournament, key=attrgetter('fitness'))

    if population.optim == 'min':
        return min(tournament, key=attrgetter('fitness'))



def roulette_wheel_selector(population):
    '''
    The whole population is considered to be selected with an associated probability
        of being picked, proportional to it's fitness.
    '''
    # Get the indices of the whole population, as np.random.choice does not work on custom classes.
    idxs = list(range(len(population)))
    indv_fitness = np.array([indv.fitness for indv in population.individuals])
    # Using numpy's vectorized operations to get an array with the proportions in the
    #   correspondent indices.
    prob_of_selection = indv_fitness / indv_fitness.sum()
    # Parameter p takes an array with the associated probability per element of the
    #   given array of being picked.
    selected_idx = np.random.choice(idxs, size=1, p=prob_of_selection)[0]
    return population.individuals[selected_idx]


def elitist_selector(population, pool_size=5):
    '''
    Similar to tournament selection, but the "contenders" are only the n best of
        the whole population. May lead to loss of diversity quickly, but may also
        converge the algorithm faster.
    '''
    elites = sorted(population, key=attrgetter('fitness'))[:pool_size]
    return choice(elites)
