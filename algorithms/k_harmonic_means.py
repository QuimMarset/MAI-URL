import numpy as np
from algorithms.distance_based_clustering import DistanceBasedClustering


class KHarmonicMeans(DistanceBasedClustering):

    def __init__(self, num_clusters, max_iterations, p, threshold, repetitions):
        super().__init__(num_clusters, max_iterations, repetitions)
        self.p = p
        self.threshold = threshold


    def compute_cluster_membership(self, distances):
        exp_distances = distances ** (self.p + 2)
        norm_factor = np.sum(exp_distances, axis=1)
        norm_factor[norm_factor == 0] += 1e-8
        pre_inverse = exp_distances / np.expand_dims(norm_factor, axis=-1)
        pre_inverse[pre_inverse == 0] += 1e-8
        membership = 1 / pre_inverse
        membership = membership / np.sum(membership, axis=1)[:, np.newaxis]
        return membership


    def compute_data_weights(self, distances):
        exp_distances = distances ** (self.p + 2)
        exp_distances[exp_distances == 0] += 1e-8
        inverse_num = 1 / exp_distances
        numerator = np.sum(inverse_num, axis=1)

        exp_distances_2 = distances ** (self.p)
        exp_distances_2[exp_distances_2 == 0] += 1e-8
        inverse_den = 1 / exp_distances_2
        denominator = np.sum(inverse_den, axis=1) ** 2

        return numerator / denominator


    def has_converged(self, centers, old_centers):
        return np.sum(np.linalg.norm(centers - old_centers)) <= self.threshold


    def compute_objective_function(self, distances):
        distances[distances == 0] += 1e-8
        inverted_distances = 1.0 / (distances ** self.p)
        denominator = np.sum(inverted_distances, axis=1)
        return np.sum(self.num_clusters / denominator)