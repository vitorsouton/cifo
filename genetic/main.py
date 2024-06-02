import time
import sys
import itertools
import pickle
import multiprocessing

from .selection.selection import tournament_sel, elitist_selector, roulette_wheel_selector
from .crossover.crossover import single_centroid_crossover, pairwise_crossover, PNN
from .mutation.mutation import coordinate_mutation, random_swap_mutation
from .algorithm.mendel import Individual, Population

from termcolor import cprint
from datetime import datetime


log_dict = {'time': multiprocessing.Manager().list(), 'loop_time': multiprocessing.Manager().list(),
            'best': multiprocessing.Manager().list(), 'selection': multiprocessing.Manager().list(),
            'xo': multiprocessing.Manager().list(), 'mutation': multiprocessing.Manager().list()
            }

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
        return None
    return wrapper


def parallel_evolve(pops, **kwargs):
    procs = []
    for pop in pops:
        proc = multiprocessing.Process(target=evolve, args=(pop,), kwargs=kwargs)
        procs.append(proc)
        proc.start()

    for p in procs:
        p.join()



def evolve(pop, **kwargs):
    start_time = time.time()
    loop_time, best = pop.evolve(**kwargs)
    end_time = time.time() - start_time

    log_dict['time'].append(end_time)
    log_dict['loop_time'].append(loop_time)
    log_dict['best'].append(best.representation)
    log_dict['selection'].append(kwargs['selection'].__name__)
    log_dict['xo'].append(kwargs['xo'].__name__)
    log_dict['mutation'].append(kwargs['mutate'].__name__)


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


        for comb in itertools.product(*combs):
            # Unpacking the functions
            selection, xo, mutation = comb

            # Generate list of Population with desired n of runs
            pops = [Population(size=35, optim='min', individual_type=Individual, n_dim=7, n_centroids=4)\
                for _ in range(5)] # For 32 cpu cores / 32gb RAM, running safely on 12~15 parallel jobs

            # Create dictionary for kwargs
            evolve_params = {
                'generations': 75, 'xo_prob': 0.95,
                'mut_prob': 0.1, 'selection': selection,
                'xo': xo, 'mutate': mutation,
                'elitism': True, 'stopping_criteria': 10
            }

            parallel_evolve(pops, **evolve_params)


        ft = time.time() - st
        cprint(f'Total time: {ft} sec.', 'red')

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H:%M:%S")
        with open(f'data/logs/{dt_string}--ALGO_SELECTION.pkl', 'wb') as f:
            for k, v in log_dict.items():
                log_dict[k] = list(v)
            pickle.dump(log_dict, f)

    if sys.argv[1] == 'run':

        # 10 loops * 5 population inits == 50 runs
        for l in range(1, 11):
            cprint(f'Starting loop {l}\n', 'yellow')
            st = time.time()

            # Should be enough to do some inference...
            pops = [Population(size=35, optim='min', individual_type=Individual, n_dim=7, n_centroids=4)\
                    for _ in range(5)] # For 32 cpu cores / 32gb RAM, running safely on 12~15 parallel jobs
                                       # Although after 5~7, using PNN, the threads are consumed

            # Best Algorithms
            selection = elitist_selector
            xo = PNN
            mutation = random_swap_mutation

            # Create dictionary for kwargs
            evolve_params = {
                'generations': 75, 'xo_prob': 0.95,
                'mut_prob': 0.1, 'selection': selection,
                'xo': xo, 'mutate': mutation,
                'elitism': True, 'stopping_criteria': 10
            }

            parallel_evolve(pops, **evolve_params)

            ft = time.time() - st
            cprint(f'Total time: {ft:.2f} sec.', 'red')

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H:%M:%S")
        with open(f'data/logs/{dt_string}--FULL_RUN.pkl', 'wb') as f:
            for k, v in log_dict.items():
                log_dict[k] = list(v)
            pickle.dump(log_dict, f)
