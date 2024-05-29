import numpy as np

def generate_mocking_data():
    return np.array([np.random.uniform(0, 1, 2) for _ in range(5)])
