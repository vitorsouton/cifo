import numpy as np

def generate_mocking_data(n_centroids=2):
    return [np.random.uniform(0, 1, 2) for _ in range(n_centroids)]
