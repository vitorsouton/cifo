import time
import sys
import itertools
import pickle


from .selection.selection import tournament_sel, elitist_selector, roulette_wheel_selector
from .crossover.crossover import single_centroid_crossover, pairwise_crossover, PNN
from .mutation.mutation import coordinate_mutation, random_swap_mutation
from .algorithm.mendel import Individual, Population

from termcolor import cprint
from datetime import datetime
from multiprocessing import Process


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


def parallel_evolve(pops, **kwargs):
    procs = []
    for pop in pops:
        proc = Process(target=evolve, args=(pop,), kwargs=kwargs)
        procs.append(proc)
        proc.start()

    for p in procs:
        p.join()

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
        n_combs = len(list(itertools.product(*combs)))

        # Multiprocess it, maybe?
        comb_procs = []
        for comb in itertools.product(*combs):

            # Unpacking the functions
            selection, xo, mutation = comb

            # Generate list of Population with desired n of runs
            pops = [Population(size=50, optim='min', individual_type=Individual, n_dim=7, n_centroids=5)\
                for _ in range(3)]

            # Create dictionary for kwargs
            evolve_params = {
                'generations': 50, 'xo_prob': 0.9,
                'mut_prob': 0.1, 'selection': selection,
                'xo': xo, 'mutate': mutation,
                'elitism': True, 'stopping_criteria':  15
            }

            # Multiprocessing, hopefully
            proc = Process(target=parallel_evolve, args=(pops,), kwargs=evolve_params)
            comb_procs.append(proc)
            proc.start()

        for p in comb_procs:
            p.join()

            n_combs -= 1

        ft = time.time() - st
        cprint(f'Total time:{ft} sec.', 'red')

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H:%M:%S")
        with open(f'../data/logs/{dt_string}.pkl', 'wb') as f:
            pickle.dump(log_dict, f)

    if sys.argv[1] == 'run':
        print('TO IMPLEMENT')
