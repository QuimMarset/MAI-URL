import numpy as np
from algorithms.distance_based_clustering import DistanceBasedClustering


class KMeans(DistanceBasedClustering):

    def __init__(self, num_clusters, max_iterations, repetitions):
        super().__init__(num_clusters, max_iterations, repetitions)


    def compute_cluster_membership(self, distances):
        num_points = distances.shape[0]
        membership = np.zeros_like(distances)
        membership[np.arange(num_points), np.argmin(distances**2, axis=1)] = 1
        return membership


    def compute_data_weights(self, distances):
        return np.ones(distances.shape[0])


    def has_converged(self, centers, old_centers):
        return np.all(centers == old_centers)


    def compute_objective_function(self, distances):
        return np.sum(np.min(distances**2, axis=0))