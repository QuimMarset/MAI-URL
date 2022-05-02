import numpy as np
from algorithms.base_clustering import BaseClustering


class FuzzyCMeans(BaseClustering):

    def __init__(self, num_clusters, max_iterations, fuzzy_degree, repetitions, is_forgy_initialization):
        super().__init__(num_clusters, max_iterations, repetitions, is_forgy_initialization)
        self.fuzzy_degree = fuzzy_degree

    def compute_cluster_membership(self, distances):
        exp_distances = distances ** (-2 / (self.fuzzy_degree - 1))
        normalization_factor = np.sum(exp_distances, axis=0)
        membership = exp_distances / normalization_factor
        membership = membership ** self.fuzzy_degree
        return membership

    def compute_data_weights(self, distances):
        return np.ones(distances.shape[1])

    def compute_objective_function(self, distances):
        membership = self.compute_cluster_membership(self, distances)
        return np.sum(membership * distances)