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

    def fit(self, data, centers_init_method=None, initial_centers=None):
        assert centers_init_method is not None or initial_centers is not None, \
            'Either the initialization method or the initial centers must be provided'

        best_centers = None
        best_membership = None
        best_qualities = None
        min_objective = None

        if initial_centers is not None:
            best_membership, best_centers, _, best_qualities = self.fit_repetition(data, initial_centers)
            
        elif centers_init_method is not None:
            for _ in range(self.repetitions):
                initial_centers = centers_init_method(self.num_clusters, data)
                membership, centers, objective_function, iteration_qualities = self.fit_repetition(data, initial_centers)

                if min_objective is None or objective_function < min_objective:
                    min_objective = objective_function
                    best_centers = centers
                    best_membership = membership
                    best_qualities = iteration_qualities
            
        self.centers = best_centers
        return best_centers, best_membership, best_qualities

    @abstractmethod
    def predict(self, data):
        pass