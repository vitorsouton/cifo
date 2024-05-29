import numpy as np

from ..utils import generate_mocking_data

def coordinate_mutation(individual):
    idx = np.random.randint(0, len(individual), 1)
    individual[idx] = np.random.uniform(0, 1)
    return individual


if __name__ == '__main__':
    coords = generate_mocking_data()
    for c in coords:
        print(f'Initial coords: {c}')
        c_mutated = coordinate_mutation(c)
        print(f'Coords after mutation: {c_mutated}')
