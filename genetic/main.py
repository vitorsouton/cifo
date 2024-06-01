import time
import sys
import itertools


from .selection.selection import tournament_sel, elitist_selector, roulette_wheel_selector
from .crossover.crossover import single_centroid_crossover, pairwise_crossover, PNN
from .mutation.mutation import coordinate_mutation, random_swap_mutation
from .algorithm.mendel import Individual, Population
from multiprocessing import Pool


log_dict = {}

def log(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        func(self, *args, **kwargs)
        end_time = time.time() - start_time
        if isinstance(log_dict, dict):
            if len(log_dict) == 0:
                log_dict['time'] = []
                log_dict['loop_time'] = []
                log_dict['best'] = []
                log_dict['selection'] = []
                log_dict['xo'] = []
                log_dict['mutation'] = []

            log_dict['time'].append(end_time)
            log_dict['loop_time'].append(self.loop_time)
            log_dict['best'].append(self.best)
            log_dict['selection'].append(kwargs['selection'].__name__)
            log_dict['xo'].append(kwargs['xo'].__name__)
            log_dict['mutation'].append(kwargs['mutate'].__name__)
    return wrapper


@log
def evolve(pop, **kwargs):
    pop.evolve(**kwargs)


if __name__ == '__main__':
    if sys.argv[1] == 'preliminary':
        st = time.time()
        # Generate all possible combinations of algorithms
        selections = [tournament_sel, elitist_selector, roulette_wheel_selector]
        xos = [single_centroid_crossover,  pairwise_crossover, PNN]
        mutations = [coordinate_mutation, random_swap_mutation]
        combs = [selections, xos, mutations]

        # Run every possible combination of algorithms
        for comb in itertools.product(*combs):
            pop = Population(
                size=50, optim='min', individual_type=Individual,
                n_dim=7, n_centroids=4
                )

            # Unpacking the functions
            selection, xo, mutation = comb

            # The actual evolving steps, separated in a function to log it
            evolve(pop,
                generations=50, xo_prob=0.9,
                mut_prob=0.1, selection=selection,
                xo=xo, mutate=mutation,
                elitism=True, stopping_criteria=15
            )

        ft = time.time() - st
        print(ft)


    if sys.argv[1] == 'run':
        print('TO IMPLEMENT')
