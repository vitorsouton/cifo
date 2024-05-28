'''
Charles Darwin had his well-deserved share of praising during the course.
This module is dedicated to Gregor Mendel (https://en.wikipedia.org/wiki/Gregor_Mendel), the father of genetics!
'''
import os

import pandas as pd

from operator import attrgetter
from random import choice, sample, random, uniform
from copy import copy
from termcolor import cprint


class Individual:
    def __init__(self, representation=None, size=None, valid_set=None, replacement=True):
        if representation is None:
            if replacement:
                self.representation = [choice(valid_set) for _ in range(size)]
            else:
                self.representation = sample(valid_set, size)
        self.representation = representation

        self.fitness = self.get_fitness()
        self.data = self.get_data()

    def get_fitness(self):
        # TODO: Implement KMeans' Inertia
        pass

    def get_data(self):
        path = os.path.join(os.path.abspath(os.path.curdir), 'data', 'clean_data', 'clean_data.csv')
        data = pd.read_csv(path)
        return data

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
        self.size = size
        self.optim = optim
        self.individual_type = individual_type
        self.individuals = []

        for _ in range(size):
            self.individuals.append(
                self.individual_type(
                    size=kwargs['sol_size'],
                    valid_set=kwargs['valid_set']
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

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]


if __name__ == '__main__':
    idv = Individual(
        # 4 centroids of 7 dimensions
        # [[7], [7], [7], [7]]
        representation=[[uniform(0, 1) for _ in range(7)] for _ in range(4)]
    )

    print(idv.representation)
    print(idv.data.columns)
