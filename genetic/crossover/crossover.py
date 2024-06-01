import numpy as np

from ..utils import generate_mocking_data
from ..algorithm.mendel import Individual
from sklearn.metrics import DistanceMetric

def single_centroid_crossover(p1, p2):
    '''
    Analogous to single point crossover
    '''
    original_shape = p1.shape
    idx = np.random.randint(0, len(p1))
    o1 = np.append(p1[:idx], p2[idx:]).reshape(original_shape)
    o2 = np.append(p2[:idx], p1[idx:]).reshape(original_shape)

    return o1, o2


def PNN(p1, p2):
    '''
    - Cross solutions:
        => Needs centroids (p1, p2) and partitions (part1, part2)
        => Combine centroids (cnew), combine partitions or recalculate it (pnew)
        => Create new centroids
        => Remove empty clusters
        => Perform PNN
    '''
    # Check PNN XO (Pairwise Neareast Neighbor) [Franti, 2000: Pattern recognition letters]

    # Combine centroids
    cnew = _combine_centroids(p1, p2)

    # Get new partition (labels according new centroid list)
    pnew = _get_new_partition(p1, p2)

    while cnew.shape[0] > p1.shape[0]:
        # Remove the centroids without points associated
        useful_cnew = _remove_empty_clusters(cnew, pnew)

        # Get centroids distance matrix
        dists = _get_weighted_centroids_distances(useful_cnew, pnew)

        # Merge closest centroids
        cnew = _merge_closest_centroids(cnew, dists)

        # Get new distances with merged centroids
        pnew = _get_new_distances(cnew, p1, p2)

    return cnew




#### Auxiliary functions for PNN ####
def _combine_centroids(p1, p2):
    return np.concatenate([p1, p2])

def _get_new_partition(p1, p2):
    dists = np.vstack([p1.distances, p2.distances])
    pnew = np.argmin(dists, axis=1)
    return pnew

def _remove_empty_clusters(cnew, pnew):
    useful_centroids = []
    for i in range(cnew.shape[0]):
        if i in pnew:
            useful_centroids.append(cnew[i])
    return np.array(useful_centroids)


def _get_weighted_centroids_distances(useful_cnew, pnew):
    # Get number of points per cluster
    unique, counts = np.unique(pnew, return_counts=True)
    ns = dict(zip(unique, counts))

    # Find nearest centroid neighbors
    dist = DistanceMetric().get_metric('minkowski')
    dists = dist.pairwise(useful_cnew, useful_cnew)

    # Generate a list of impacts of merging centroids (Ward's) (na*nb / na+nb)
    impacts = [[] for _ in range(max(ns.keys())+1)]
    for k, v in ns.items():
        for vv in ns.values():
            impacts[k].append((v * vv) / (v + vv))

    # Remove empty lists
    impacts = list(filter(lambda x: len(x) > 0, impacts))
    impacts = np.array(impacts)

    # Mulptiply the distances by the impact value of merging
    dists = impacts * dists
    return dists

def _merge_closest_centroids(cnew, dists):
    to_merge_idx = np.where(dists==np.min(dists[np.nonzero(dists)]))[0]
    merged = [cnew[to_merge_idx].mean(axis=0)]
    cnew = np.delete(cnew, to_merge_idx, axis=0)
    cnew = np.append(cnew, merged, axis=0)
    return cnew

def _get_new_distances(cnew, p1, p2):
    ps = np.vstack([p1.data, p2.data])
    dist = DistanceMetric().get_metric('minkowski')
    dists = dist.pairwise(ps, cnew)
    pnew = np.argmin(dists, axis=1)
    return pnew




if __name__ == '__main__':
    data = generate_mocking_data(n_points=20)
    p1 = Individual(data=data, n_dim=2, n_centroids=3)
    p2 = Individual(data=data, n_dim=2, n_centroids=3)

    print(PNN(p1, p2))
