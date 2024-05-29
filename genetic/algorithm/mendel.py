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
from ..selection.selection import tournament_sel
from ..crossover.crossover import single_coordinate_crossover
from ..mutation.mutation import coordinate_mutation
from time import time


class Individual:
    def __init__(self, data, n_dim=7, n_centroids=4, representation=None):
        if representation is None:
            self.representation = np.array([np.random.uniform(0, 1, n_dim) for _ in range(n_centroids)])
        else:
            self.representation = representation

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

    def __lt__(self, other):
        return self.fitness < other.fitness



class Population:
    def __init__(self, size, optim, individual_type, **kwargs):
        self.size = size
        self.optim = optim
        self.individual_type = individual_type
        self.individuals = []
        self.data = self.get_data()
        self.loop_time = []


        for _ in range(size):
            self.individuals.append(
                self.individual_type(
                    n_dim=kwargs['n_dim'],
                    n_centroids=kwargs['n_centroids'],
                    data=self.data
                )
            )

    def log_time(func):
        def wrapper(self, *args, **kwargs):
            start_time = time()
            func(self, *args, **kwargs)
            end_time = time() - start_time
            print(f'Total time of execution: {end_time:.2f} seconds.')
            print(f'Average time per generation: {np.mean(self.loop_time): .2f} seconds.')
        return wrapper

    @log_time
    def evolve(self, generations, xo_prob, mut_prob, selection, xo, mutate, elitism, stopping_criteria):
        if self.optim == 'max':
            best = max(self.individuals, key=attrgetter('fitness'))
        elif self.optim == 'min':
            best = min(self.individuals, key=attrgetter('fitness'))

        print(f'Initial fitness: {best}')
        early_stopping = stopping_criteria # Patience for breaking condition

        for g in range(generations):
            loop_time_start = time()

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

                new_pop.append(self.individual_type(representation=off1, data=self.data))
                if len(new_pop) < self.size:
                    new_pop.append(self.individual_type(representation=off2, data=self.data))

            self.individuals = new_pop

            best_ind_generation = min(self, key=attrgetter("fitness"))
            if best_ind_generation < best:
                best = best_ind_generation
                early_stopping = stopping_criteria
            else:
                early_stopping -= 1

            loop_time_end = time() - loop_time_start
            self.loop_time.append(loop_time_end)

            if early_stopping == 0:
                cprint(f'Stop improving after {g} generations.', 'red')
                break

        cprint(f'Best solution found: {min(self, key=attrgetter("fitness"))}', 'green')

    def get_data(self):
        path = os.path.join(os.path.abspath(os.path.curdir), 'data', 'clean_data', 'clean_data.csv')
        data = pd.read_csv(path)
        return data


    def __repr__(self):
        return f'Population of {type(self.individuals[0])}'

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]


if __name__ == '__main__':
    pop = Population(
        size=100, optim='min', individual_type=Individual,
        n_dim=7, n_centroids=4
    )

    pop.evolve(
        generations=100, xo_prob=0.9,
        mut_prob=0.05, selection=tournament_sel,
        xo=single_coordinate_crossover, mutate=coordinate_mutation,
        elitism=True, stopping_criteria=10
    )
