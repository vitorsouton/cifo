import numpy as np

from ..utils import generate_mocking_data

def single_coordinate_crossover(p1, p2):
    idx1, idx2 = np.random.randint(0, len(p1), 2)
    p1[idx1], p2[idx2] = p2[idx2], p1[idx1]
    return p1, p2


if __name__ == '__main__':
    centroids = generate_mocking_data()
    p1, p2 = centroids[0], centroids[1]
    print(f'Parents: {p1} and {p2}')
    o1, o2 = single_coordinate_crossover(p1, p2)
    print(f'Offspring: {o1} and {o2}')
