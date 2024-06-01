from .selection.selection import tournament_sel, elitist_selector
from .crossover.crossover import single_centroid_crossover, pairwise_nearest_neighbor_crossover
from .mutation.mutation import coordinate_mutation
from .algorithm.mendel import Individual, Population


if __name__ == '__main__':
    pop = Population(
    size=25, optim='min', individual_type=Individual,
    n_dim=7, n_centroids=4
    )

    pop.evolve(
        generations=50, xo_prob=0.9,
        mut_prob=0.05, selection=elitist_selector,
        xo=pairwise_nearest_neighbor_crossover, mutate=coordinate_mutation,
        elitism=True, stopping_criteria=10
    )
