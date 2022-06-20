from math import dist
from cv2 import threshold
import numpy as np
from algorithms.distance_based_clustering import DistanceBasedClustering


class FuzzyCMeans(DistanceBasedClustering):

    def __init__(self, num_clusters, max_iterations, fuzzy_degree, threshold, repetitions):
        super().__init__(num_clusters, max_iterations, repetitions)
        self.fuzzy_degree = fuzzy_degree
        self.threshold = threshold


    def compute_cluster_membership(self, distances):
        exp_distances = distances ** (2 / (self.fuzzy_degree - 1))
        norm_factor = np.sum(exp_distances, axis=1)
        norm_factor[norm_factor == 0] += 1e-8
        pre_inverse = exp_distances / np.expand_dims(norm_factor, axis=-1)
        pre_inverse[pre_inverse == 0] += 1e-8
        membership = 1 / pre_inverse
        membership = membership / np.sum(membership, axis=1)[:, np.newaxis]
        return membership**self.fuzzy_degree


    def compute_data_weights(self, distances):
        return np.ones(distances.shape[0])


    def has_converged(self, centers, old_centers):
        return np.sum(np.linalg.norm(centers - old_centers)) <= self.threshold


    def compute_objective_function(self, distances):
        membership = self.compute_cluster_membership(distances)
        return np.sum(membership * distances**2)