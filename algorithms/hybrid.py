import numpy as np
from algorithms.k_harmonic_means import KHarmonicMeans


class Hybrid1(KHarmonicMeans):

    def compute_cluster_membership(self, distances):
        num_points = distances.shape[1]
        membership = np.zeros((self.num_clusters, num_points))
        membership[np.arange(self.num_clusters), np.argmin(distances, axis=0)] = 1
        return membership


    def compute_data_weights(self, distances):
        numerator = np.sum(distances ** (-self.p - 2), axis=0)
        denominator = np.sum(distances ** (-self.p)) ** 2
        return numerator / denominator


class Hybrid2(KHarmonicMeans):

    def compute_cluster_membership(self, distances):
        numerator = distances ** (-self.p - 2)
        denominator = np.sum(numerator, axis=0)
        return numerator / denominator


    def compute_data_weights(self, distances):
        return np.ones(distances.shape[1])