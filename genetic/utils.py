import numpy as np

def generate_mocking_data(n_points=5, n_centroids=2):
    return np.array([np.random.uniform(0, 1, n_centroids) for _ in range(n_points)])
