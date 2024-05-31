import numpy as np

from ..utils import generate_mocking_data

def single_centroid_crossover(p1, p2):
    '''
    Analogous to single point crossover
    '''
    original_shape = p1.shape
    idx = np.random.randint(0, len(p1))
    o1 = np.append(p1[:idx], p2[idx:]).reshape(original_shape)
    o2 = np.append(p2[:idx], p1[idx:]).reshape(original_shape)

    return o1, o2


# Check PNN XO (Pairwise Neareast Neighbor)



# KMeans to fine tune after crossover?


if __name__ == '__main__':
    p1 = generate_mocking_data()
    p2 = generate_mocking_data()

    print(f'Parents: {p1} and {p2}')
    o1, o2 = single_centroid_crossover(p1, p2)
    print(f'Offspring: {o1} and {o2}')
