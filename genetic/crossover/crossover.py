import numpy as np

from ..utils import generate_mocking_data
from ..algorithm.mendel import Individual
from sklearn.metrics import DistanceMetric
from sklearn.cluster import KMeans

def single_centroid_crossover(p1, p2):
    '''
    Analogous to single point crossover
    '''
    original_shape = p1.shape
    idx = np.random.randint(0, len(p1))
    o1 = np.append(p1[:idx], p2[idx:]).reshape(original_shape)
    o2 = np.append(p2[:idx], p1[idx:]).reshape(original_shape)

    return o1, o2


def pairwise_crossover(p1, p2):
    '''
    Take 2 parents, combine their representations, assign the points to them,
        remove the representations without assigned points, merge the representations
        which are closest, repeat until the length or the merged == length of representation.
    The detailed description of each step will be written in the respective function.
    '''
    # Check PNN XO (Pairwise Neareast Neighbor) [Franti, 2000: Pattern recognition letters]
    # https://www.sciencedirect.com/science/article/abs/pii/S0167865599001336

    # Combine centroids
    cnew = _combine_centroids(p1, p2)

    # Get new partition (labels according new centroid list)
    pnew = _get_new_partition(p1, p2)

    # Assures representation consistency
    while cnew.shape[0] > p1.shape[0]:
        # Remove the centroids without points associated
        cnew = _remove_empty_clusters(cnew, pnew)

        # Get centroids distance matrix
        dists = _get_weighted_centroids_distances(cnew, pnew)

        # Merge closest centroids
        cnew = _merge_closest_centroids(cnew, dists)

        # Get new distances with merged centroids
        pnew = _get_new_distances(cnew, p1)

    return cnew


def PNN(p1, p2):
    '''
    Same as pairwise, with extra KMeans step before merging.
    '''
    # Check PNN XO (Pairwise Neareast Neighbor) [Franti, 2000: Pattern recognition letters]
    # https://www.sciencedirect.com/science/article/abs/pii/S0167865599001336

    # Combine centroids
    cnew = _combine_centroids(p1, p2)

    # Get new partition (labels according new centroid list)
    pnew = _get_new_partition(p1, p2)

    # Assures representation consistency
    while cnew.shape[0] > p1.shape[0]:
        # Remove the centroids without points associated
        cnew = _remove_empty_clusters(cnew, pnew)

        # Adjust centroids using KMeans
        cnew = _adjust_centroids(cnew, p1)

        # Get centroids distance matrix
        dists = _get_weighted_centroids_distances(cnew, pnew)

        # Merge closest centroids
        cnew = _merge_closest_centroids(cnew, dists)

        # Get new distances with merged centroids
        pnew = _get_new_distances(cnew, p1)

    return cnew




#### Auxiliary functions for pairwise and PNN ####

def _combine_centroids(p1, p2):
    '''
    Takes as input the parents and outputs the UNION of their representation.
        The output is a new meta-representation with n*2 centroids.
    '''
    return np.concatenate([p1, p2])

def _get_new_partition(p1, p2):
    '''
    Given the new meta-representation, returns a matrix with the new labels associated to it.
    '''
    # We have the distances pre-calculated and saved in a attribute of each parent.
    #   We simply stack the distances columnwise and take the min ones.
    dists = np.hstack([p1.distances, p2.distances])
    pnew = np.argmin(dists, axis=1)
    return pnew

def _remove_empty_clusters(cnew, pnew):
    '''
    If we have clusters in the meta-representation with no associated points, we cut them off.
    '''
    useful_centroids = []
    for i in range(cnew.shape[0]):
        if i in pnew:
            useful_centroids.append(cnew[i])
    return np.array(useful_centroids)


def _get_weighted_centroids_distances(useful_cnew, pnew):
    '''
    Calculate the distance between the centroids and multiply it by a measure of
        the associated points.
    '''
    # Get number of points per cluster
    unique, counts = np.unique(pnew, return_counts=True)
    ns = dict(zip(unique, counts))

    # Find nearest centroid neighbors
    dist = DistanceMetric().get_metric('euclidean')
    dists = dist.pairwise(useful_cnew, useful_cnew)

    # Generate a list of impacts of merging centroids  (na*nb / na+nb)
    impacts = [[] for _ in range(max(ns.keys())+1)]
    for k, v in ns.items():
        for vv in ns.values():
            impacts[k].append((v * vv) / (v + vv))

    # Remove empty lists
    #   The previous step may generate empty lists for empty centroids.
    #   To account for it, we simply remove them.
    impacts = list(filter(lambda x: len(x) > 0, impacts))
    impacts = np.array(impacts)

    # Mulptiply the distances by the impact value of merging
    dists = impacts * dists
    return dists

def _merge_closest_centroids(cnew, dists):
    '''
    Finds the 2 closest centroids using the weighted distance matrix and merges
        them into a single one, with coordinates given by the mean.
    '''
    try:
        to_merge_idx = np.where(dists==np.min(dists[np.nonzero(dists)]))[0]
    except ValueError:
        to_merge_idx = np.random.randint(0, cnew.shape[0], 2)

    merged = [cnew[to_merge_idx].mean(axis=0)]
    cnew = np.delete(cnew, to_merge_idx, axis=0)
    cnew = np.append(cnew, merged, axis=0)
    return cnew


def _get_new_distances(cnew, p1):
    '''
    Calculates the new labels associated with the new set of merged clusters.
    '''
    dist = DistanceMetric().get_metric('euclidean')
    dists = dist.pairwise(p1.data, cnew)
    pnew = np.argmin(dists, axis=1)
    return pnew


def _adjust_centroids(cnew, p1):
    '''
    For the PNN algorithm.
    Before getting the distance, a single iteration of KMeans algorithm is performed.
    This "pulls" the centroids a little bit more to the density center of their closest cluster.
    '''
    n_clusters = cnew.shape[0]
    kmeans = KMeans(n_clusters, init=cnew, n_init=1, max_iter=2)
    kmeans.fit(p1.data)
    return kmeans.cluster_centers_


if __name__ == '__main__':
    data = generate_mocking_data(n_points=20)
    p1 = Individual(data=data, n_dim=2, n_centroids=3)
    p2 = Individual(data=data, n_dim=2, n_centroids=3)

    print(pairwise_crossover(p1, p2))
