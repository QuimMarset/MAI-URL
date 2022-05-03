import numpy as np
from algorithms.distance_based_clustering import DistanceBasedClustering


class KMeans(DistanceBasedClustering):

    def __init__(self, num_clusters, max_iterations, repetitions, is_forgy_initialization):
        super().__init__(num_clusters, max_iterations, is_forgy_initialization)
        self.repetitions = repetitions


    def compute_cluster_membership(self, distances):
        num_points = distances.shape[1]
        membership = np.zeros((self.num_clusters, num_points))
        membership[np.arange(self.num_clusters), np.argmin(distances, axis=0)] = 1
        return membership


    def compute_data_weights(self, distances):
        return np.ones(distances.shape[1])


    def has_converged(self, centers, old_centers):
        return np.any(centers == old_centers)


    def compute_objective_function(self, distances):
        return np.sum(np.min(distances, axis=0))