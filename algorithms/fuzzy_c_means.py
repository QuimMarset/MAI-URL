from cv2 import threshold
import numpy as np
from algorithms.distance_based_clustering import DistanceBasedClustering


class FuzzyCMeans(DistanceBasedClustering):

    def __init__(self, num_clusters, max_iterations, fuzzy_degree, threshold, repetitions, is_forgy_initialization):
        super().__init__(num_clusters, max_iterations, repetitions, is_forgy_initialization)
        self.fuzzy_degree = fuzzy_degree
        self.repetitions = repetitions
        self.threshold = threshold


    def compute_cluster_membership(self, distances):
        exp_distances = distances ** (-2 / (self.fuzzy_degree - 1))
        normalization_factor = np.sum(exp_distances, axis=0)
        membership = exp_distances / normalization_factor
        membership = membership ** self.fuzzy_degree
        return membership


    def compute_data_weights(self, distances):
        return np.ones(distances.shape[1])


    def has_converged(self, centers, old_centers):
        return np.sum(np.linalg.norm(centers - old_centers)) <= self.threshold


    def compute_objective_function(self, distances):
        membership = self.compute_cluster_membership(self, distances)
        return np.sum(membership * distances)