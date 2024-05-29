'''
Charles Darwin had his well-deserved share of praising during the course.
This module is dedicated to Gregor Mendel (https://en.wikipedia.org/wiki/Gregor_Mendel), the father of genetics!
'''
import os

import pandas as pd
import numpy as np

from operator import attrgetter
from random import random
from copy import copy
from termcolor import cprint
from sklearn.metrics import DistanceMetric
from ..selection import tournament_sel


class Individual:
    def __init__(self, data, n_dim=7, n_centroids=4):
        self.representation = np.array([np.random.uniform(0, 1, n_dim) for _ in range(n_centroids)])
        self.labels = None
        self.data = data
        self.fitness = self.get_fitness()

    def get_fitness(self):
        dist = DistanceMetric.get_metric('minkowski')
        distances = dist.pairwise(self.data, self.representation) # To all centroids
        self.labels = np.argmin(distances, axis=1) # Labels
        # Distance to label centroids
        fitness = np.min(distances, axis=1).sum() # INERTIA!
        return fitness

        #TODO: If we have time, try to implement different fitness functions
        #      to try and make GA better than KNN


    def __repr__(self):
        return f'Fitness: {self.fitness}'

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, idx):
        return self.representation[idx]

    def __setitem__(self, idx, value):
        self.representation[idx] = value


class Population:
    def __init__(self, size, optim, individual_type, **kwargs):
        self.optim = optim
        self.individual_type = individual_type
        self.individuals = []
        self.data = self.get_data()


        for _ in range(size):
            self.individuals.append(
                self.individual_type(
                    n_dim=kwargs['n_dim'],
                    n_centroids=kwargs['n_centroids'],
                    data=kwargs['data']
                )
            )


    def evolve(self, generations, xo_prob, mut_prob, selection, xo, mutate, elitism):
        if self.optim == 'max':
            best = max(self.individuals, key=attrgetter('fitness'))
        elif self.optim == 'min':
            best = min(self.individuals, key=attrgetter('fitness'))

        print(f'Initial fitness: {best}')

        for _ in range(generations):
            new_pop = []

            if elitism:
                if self.optim == "max":
                    elite = copy(max(self.individuals, key=attrgetter('fitness')))
                elif self.optim == "min":
                    elite = copy(min(self.individuals, key=attrgetter('fitness')))

                new_pop.append(elite)

            while len(new_pop) < self.size:
                # Selection step
                parent1, parent2 = selection(self), selection(self)

                # Crossover step
                if random() < xo_prob:
                    off1, off2 = xo(parent1, parent2)
                else:
                    off1, off2 = parent1, parent2

                # Mutation step
                if random() < mut_prob:
                    off1 = mutate(off1)
                if random() < mut_prob:
                    off2 = mutate(off2)

                new_pop.append(self.individual_type(representation=off1))
                if len(new_pop) < self.size:
                    new_pop.append(self.individual_type(representation=off2))

            self.individuals = new_pop

        cprint(f'Best solution found: {max(self, key=attrgetter("fitness"))}', 'green')

    def get_data(self):
        path = os.path.join(os.path.abspath(os.path.curdir), 'data', 'clean_data', 'clean_data.csv')
        data = pd.read_csv(path)
        return data

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]


if __name__ == '__main__':
    path = os.path.join(os.path.abspath(os.path.curdir), 'data', 'clean_data', 'clean_data.csv')
    data = pd.read_csv(path)

    idv = Individual(
        # 4 centroids of 7 dimensions
        # [[7], [7], [7], [7]]
        data=data,
        n_centroids=4,
        n_dim=7
    )

    print(idv.representation)
    print(idv)
