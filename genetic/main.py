import numpy as np

from .selection.selection import tournament_sel, elitist_selector, roulette_wheel_selector
from .crossover.crossover import single_centroid_crossover, pairwise_crossover, PNN
from .mutation.mutation import coordinate_mutation, random_swap_mutation
from .algorithm.mendel import Individual, Population


if __name__ == '__main__':
    pop = Population(
    size=50, optim='min', individual_type=Individual,
    n_dim=7, n_centroids=4
    )

    pop.evolve(
        generations=50, xo_prob=0.9,
        mut_prob=0.1, selection=elitist_selector,
        xo=PNN, mutate=random_swap_mutation,
        elitism=True, stopping_criteria=15
    )
