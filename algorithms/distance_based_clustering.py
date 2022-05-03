from algorithms.base_clustering import BaseClustering
import numpy as np
from abc import abstractmethod


class DistanceBasedClustering(BaseClustering):

    def __init__(self, num_clusters, max_iterations, is_forgy_initialization):
        super().__init__(num_clusters, max_iterations, is_forgy_initialization)


    @abstractmethod
    def compute_cluster_membership(self, distances):
        pass


    @abstractmethod
    def compute_data_weights(self, distances):
        pass


    @abstractmethod
    def has_converged(self, centers, old_centers):
        pass


    @abstractmethod
    def compute_objective_function(self, distances):
        pass


    def fit(self, data):
        self.centers = self.initialize_cluster_centers(data)
        iteration = 0
        has_converged = False

        while iteration < self.max_iterations and not has_converged:
            old_centers = self.centers.copy()
            
            distances = np.linalg.norm(data - np.expand_dims(self.centers, axis=1), axis=2)**2
            membership = self.compute_cluster_membership(distances)
            weights = self.compute_data_weights(distances)
            self.centers = self.recompute_center_location(data, membership, weights)
            
            iteration += 1
            has_converged = self.has_converged(self.centers, old_centers)
        
        objective_function = self.compute_objective_function(distances)
        distances = np.linalg.norm(data - np.expand_dims(self.centers, axis=1), axis=2)**2
        membership = self.compute_cluster_membership(distances)
        return membership, self.centers, objective_function


    def predict(self, data):
        distances = np.linalg.norm(data - np.expand_dims(self.centers, axis=1), axis=2)**2
        return self.compute_cluster_membership(distances)