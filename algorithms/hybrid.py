import numpy as np
from algorithms.k_harmonic_means import KHarmonicMeans


class Hybrid1(KHarmonicMeans):

    def compute_cluster_membership(self, distances):
        num_points = distances.shape[0]
        membership = np.zeros_like(distances)
        membership[np.arange(num_points), np.argmin(distances, axis=1)] = 1
        return membership


class Hybrid2(KHarmonicMeans):

    def compute_data_weights(self, distances):
        return np.ones(distances.shape[0])