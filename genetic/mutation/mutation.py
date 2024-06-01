import numpy as np

from ..utils import generate_mocking_data

def coordinate_mutation(individual):
    idx = np.random.randint(0, len(individual), 1)
    individual[idx] = np.random.uniform(0, 1)
    return individual

def random_swap_mutation(individual):
    idx_to_swap = np.random.randint(0, len(individual), 2)
    individual[idx_to_swap[0]], individual[idx_to_swap[1]] = individual[idx_to_swap[1]], individual[idx_to_swap[0]]
    return individual

if __name__ == '__main__':
    coords = generate_mocking_data(n_centroids=3)
    for c in coords:
        print(f'Initial coords: {c}')
        c_mutated = random_swap_mutation(c)
        print(f'Coords after mutation: {c_mutated}')
