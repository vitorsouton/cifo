import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import DistanceMetric

def generate_mocking_data(n_points=5, n_centroids=2):
    return np.array([np.random.uniform(0, 1, n_centroids) for _ in range(n_points)])


def calculate_inertia(centroids, data):
    dist = DistanceMetric.get_metric('euclidean')
    distances = dist.pairwise(data, centroids) # To all centroids
    dists = np.square(distances)
    inertia = np.min(dists, axis=1).sum() # INERTIA!
    return inertia


def get_cluster_distribution(centroids, data):
    data = data.copy()
    kmeans = KMeans(n_clusters=centroids.shape[0], init=centroids,n_init=1)
    kmeans.fit(data)
    data.loc[:, 'labels'] = kmeans.predict(data)
    return data.labels.value_counts(normalize=True)
