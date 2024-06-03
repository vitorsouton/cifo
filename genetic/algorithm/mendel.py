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
from sklearn.metrics import DistanceMetric, calinski_harabasz_score
from time import time


class Individual:
    '''
    The Individual class is mostly unchanged from the labs' charles module.
    The relevant changes will be commented in the code.
    '''
    def __init__(self, data, n_dim=7, n_centroids=4, representation=None):
        if representation is None:
            # The representation consists of an numpy array of shape (n_centroids, n_dimensions).
            self.representation = np.array([np.random.uniform(0, 1, n_dim) for _ in range(n_centroids)])
        else:
            self.representation = representation

        self.labels = None
        self.distances = None
        self.data = data
        self.inertia = self.get_inertia()
        self.fitness = self.get_fitness()
        self.shape = self.representation.shape

    def get_fitness(self):
        '''
        In the beggining, the fitness function was KMeans Inertia in order to readily compare
            the GA to KMeans. In the current iteration, we decided to penalize representations
            that have that are too dispersed.
        For that, we decided to use the Calinski-Harabasz score, that measures  the ratio of
            the within-cluster variance to the between-cluster variance.
        '''

        # Add penalty to fitness. The magnitude of ch_score is higher than the inertia itself
        #   and has exponential character. To solve it, we took the ln of the score and
        #   divided the fitness. Higher scores will give lower fitness.
        ch_score = calinski_harabasz_score(self.data, self.labels)

        fitness = self.inertia
        fitness /= np.log(ch_score)


        return fitness


    def get_inertia(self):
        '''
        Formula: sum(||point - centroid||)**2
        '''
        dist = DistanceMetric.get_metric('euclidean')
        self.distances = dist.pairwise(self.data, self.representation) # To all centroids
        self.labels = np.argmin(self.distances, axis=1) # Labels

        # Distance to label centroids
        dists = np.square(self.distances)
        inertia = np.min(dists, axis=1).sum() # INERTIA!
        return inertia


    def __repr__(self):
        return f'Fitness: {self.fitness}; Inertia {self.inertia}'

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, idx):
        return self.representation[idx]

    def __setitem__(self, idx, value):
        self.representation[idx] = value

    def __lt__(self, other):
        return self.fitness < other.fitness



class Population:
    '''
    The main difference from the charles module from the lab is that the individual
        class is passed as an argument to the init and each one of them is initialized
        with the Population definition.
    '''
    def __init__(self, size, optim, individual_type, **kwargs):
        self.size = size
        self.optim = optim
        self.individual_type = individual_type
        self.individuals = []
        self.data = self.get_data() # Loads the .csv needed to calculate the fitness
        self.loop_time = []
        self.best = None
        self.history = []


        for _ in range(size):
            self.individuals.append(
                self.individual_type(
                    n_dim=kwargs['n_dim'],
                    n_centroids=kwargs['n_centroids'],
                    representation=kwargs.get('representation', None),
                    data=self.data
                )
            )


    def evolve(self, generations, xo_prob, mut_prob, selection, xo, mutate, elitism, stopping_criteria):
        '''
        The algorithm responsible for the evolution. A few changes were made from the
            lab's charles module and, other than the prints for logs, will be point out.
        '''
        if self.optim == 'max':
            self.best = max(self.individuals, key=attrgetter('fitness'))
        elif self.optim == 'min':
            self.best = min(self.individuals, key=attrgetter('fitness'))

        cprint(f'Initial: {self.best}', 'blue')

        # We decided to create an "early stopping" logic in order to cut short runs that
        #   got stuck in local minima and were not improving.
        early_stopping = stopping_criteria

        for g in range(generations):
            loop_time_start = time()

            new_pop = []
            self.history.append(self.best.fitness)

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
                    offs = xo(parent1, parent2)
                else:
                    offs = (parent1, parent2)

                # Some of the crossover algorithms returned only 1 offspring.
                #   Therefore, we had to account for it in this step.
                if isinstance(offs, tuple):
                    off1, off2 = offs
                else:
                    off1 = offs
                    off2 = None

                # Mutation step
                if random() < mut_prob:
                    off1 = mutate(off1)
                if off2 is not None:
                    if random() < mut_prob:
                        off2 = mutate(off2)

                new_pop.append(self.individual_type(representation=off1, data=self.data))
                if len(new_pop) < self.size and off2 is not None:
                    new_pop.append(self.individual_type(representation=off2, data=self.data))

            self.individuals = new_pop

            # Early stopping check
            # The logic is simple:
            # Record best individual of the generation
            best_ind_generation = min(self, key=attrgetter("fitness"))
            # If the best individual of the generation is better than the elite,
            #   we assign it as the new elite and reset the counter.
            if best_ind_generation < self.best:
                self.best = best_ind_generation
                early_stopping = stopping_criteria
            # Else, we decrement the countdown.
            else:
                early_stopping -= 1

            loop_time_end = time() - loop_time_start
            self.loop_time.append(loop_time_end)

            # If the coountdown reaches 0, the evolution stops.
            if early_stopping == 0:
                cprint(f'Stop improving after {g-stopping_criteria+1} generations.', 'red')
                break

        cprint(f'Best solution found: {self.best}\n', 'green')
        return self.loop_time, self.best


    def get_data(self):
        '''
        Simple function to load the data needed to calculate the fitness.
        '''
        path = os.path.join(os.path.abspath(os.path.curdir), 'data', 'clean_data', 'clean_data.csv')
        data = pd.read_csv(path)
        return data


    def __repr__(self):
        return f'Population of {type(self.individuals[0])}'

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]
