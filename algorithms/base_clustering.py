from abc import ABC, abstractmethod
import numpy as np


class BaseClustering(ABC):

    def __init__(self, num_clusters, max_iterations):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations

    def recompute_center_location(self, data, membership, weights):
        # Equation 1 in the paper
        membership = membership.T
        normalization_factors = np.expand_dims(membership @ weights, axis=1)
        unnormalized_centers = (membership * weights) @ data
        centers = unnormalized_centers / (normalization_factors + 1e-8)
        return centers

    @abstractmethod
    def fit(self, data, initial_centers):
        pass

    @abstractmethod
    def predict(self, data):
        pass