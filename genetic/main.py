'''
In this python script is where the magic happens.

There are 2 main ways to run it "preliminary" and "run", passed as arguments in the
    terminal.

The "preliminary" tests all the 18 possible combinations of selection, xo and mutation
    algorithms, in batches of 5 population initializations, each with 35 individuals.

The "run" loops over 10 times batches of 5 population initializations, each with 35
    individuals, using the given selection, xo and mutation triad.

I assure that everything that could be parallelized was parallelized. But given the
    sequential nature of the algorithm itself, it was a challenge.

The next step in that mission, was to try and implement the algorithm in pytorch or
    CUDA, where we can have a fine-grained control over what to run in a GPU or in
    the CPU. This should speed up some operations.
'''

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

# Creation of a data structure to log and later analyze the runs
log_dict = {'time': multiprocessing.Manager().list(), 'loop_time': multiprocessing.Manager().list(),
            'best': multiprocessing.Manager().list(), 'selection': multiprocessing.Manager().list(),
            'xo': multiprocessing.Manager().list(), 'mutation': multiprocessing.Manager().list(),
            'fitness': multiprocessing.Manager().list(), 'inertia': multiprocessing.Manager().list()
            }


def parallel_evolve(pops, **kwargs):
    '''
    The kind of "wrapper" function to run the evolution on multiple population pools
        in parallel.
    '''
    # Create a list to store the processes calls
    procs = []
    for pop in pops:
        # Create 1 process for each population pool
        proc = multiprocessing.Process(target=evolve, args=(pop,), kwargs=kwargs)
        procs.append(proc)
        # Start the process
        proc.start()

    # Fetch the result of each process and close it
    for p in procs:
        p.join()



def evolve(pop, **kwargs):
    '''
    The kind of "wrapper" function to provide logging capability to the evolution task.
    Also, it has to be done in a separated dedicated function, rather than in a Python's
        decorator, because multiprocessing messes up the decorator's functionalities.
    '''
    start_time = time.time()
    loop_time, best = pop.evolve(**kwargs)
    end_time = time.time() - start_time

    log_dict['time'].append(end_time)
    log_dict['loop_time'].append(loop_time)
    log_dict['best'].append(best.representation)
    log_dict['fitness'].append(best.fitness)
    log_dict['inertia'].append(best.inertia)

    # This last block logs the name of the variable, which will be the functions
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
        n_combs = len(list(itertools.product(*combs)))

        # Run every possible combination of algorithms
        for comb in itertools.product(*combs):
            # Unpacking the functions
            selection, xo, mutation = comb

            # Generate list of Population with desired n of runs
            # For 32 cpu cores / 32gb RAM, one can run safely 12~15 parallel jobs
            #   but there is a huge bottleneck, which is whenever PNN is the xo
            #   algorithm. The KMeans step consumes the whole threading capabilty
            #   of the CPU.
            pops = [Population(size=35, optim='min', individual_type=Individual, n_dim=7, n_centroids=4)\
                for _ in range(5)]

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

        # For logging purposes, after the run is completed, the log dictionary is
        #   saved as an pickle object with the date and time of completion
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
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
                    for _ in range(5)]

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
        dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        with open(f'data/logs/{dt_string}--FULL_RUN.pkl', 'wb') as f:
            for k, v in log_dict.items():
                log_dict[k] = list(v)
            pickle.dump(log_dict, f)
