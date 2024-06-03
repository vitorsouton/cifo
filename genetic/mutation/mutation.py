import numpy as np

from ..utils import generate_mocking_data

def coordinate_mutation(individual):
    '''
    The first and most simple idea was to randomly choose a coordinate from the
        centroid and randomly set it to another value.
    '''
    idx = np.random.randint(0, len(individual), 1)
    individual[idx] = np.random.uniform(0, 1)
    return individual

def random_swap_mutation(individual):
    '''
    This implementation is a 2-gene mutation. It randomly chooses 2 genes in the same
        individual and swap them.
    '''
    idx_to_swap = np.random.randint(0, len(individual), 2)
    individual[idx_to_swap[0]], individual[idx_to_swap[1]] = individual[idx_to_swap[1]], individual[idx_to_swap[0]]
    return individual

if __name__ == '__main__':
    coords = generate_mocking_data(n_centroids=3)
    for c in coords:
        print(f'Initial coords: {c}')
        c_mutated = random_swap_mutation(c)
        print(f'Coords after mutation: {c_mutated}')
