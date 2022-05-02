import numpy as np
from algorithms.base_clustering import BaseClustering


class KHarmonicMeans(BaseClustering):

    def __init__(self, num_clusters, max_iterations, p, repetitions, is_forgy_initialization):
        super().__init__(num_clusters, max_iterations, repetitions, is_forgy_initialization)
        self.p = p

    def compute_cluster_membership(self, distances):
        numerator = distances ** (-self.p - 2)
        denominator = np.sum(numerator, axis=0)
        return numerator / denominator

    def compute_data_weights(self, distances):
        numerator = np.sum(distances ** (-self.p - 2), axis=0)
        denominator = np.sum(distances ** (-self.p)) ** 2
        return numerator / denominator

    def compute_objective_function(self, distances):
        inverted_distances = 1.0 / (distances ** self.p)
        denominators = np.sum(inverted_distances, axis=0)
        return np.sum(self.num_clusters / denominators)