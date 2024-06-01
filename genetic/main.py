import numpy as np

from .selection.selection import tournament_sel, elitist_selector
from .crossover.crossover import single_centroid_crossover, pairwise_nearest_neighbor_crossover
from .mutation.mutation import coordinate_mutation, random_swap_mutation
from .algorithm.mendel import Individual, Population


if __name__ == '__main__':
    pop = Population(
    size=50, optim='min', individual_type=Individual,
    n_dim=7, n_centroids=4
    )

    pop.evolve(
        generations=50, xo_prob=0.9,
        mut_prob=0.1, selection=tournament_sel,
        xo=pairwise_nearest_neighbor_crossover, mutate=random_swap_mutation,
        elitism=True, stopping_criteria=15
    )
