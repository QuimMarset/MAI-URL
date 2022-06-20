from algorithms.base_clustering import BaseClustering
from abc import abstractmethod
from scipy.spatial.distance import cdist
from utils.metrics import compute_quality_metric


class DistanceBasedClustering(BaseClustering):

    def __init__(self, num_clusters, max_iterations, repetitions):
        super().__init__(num_clusters, max_iterations)
        self.repetitions = repetitions


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

    def fit_repetition(self, data, initial_centers):
        centers = initial_centers
        iteration = 0
        has_converged = False
        # Used in the second experiment
        iteration_qualities = []

        while iteration < self.max_iterations and not has_converged:
            old_centers = centers.copy()
            
            distances = cdist(data, centers)
            membership = self.compute_cluster_membership(distances)
            weights = self.compute_data_weights(distances)
            centers = self.recompute_center_location(data, membership, weights)
            
            iteration += 1
            has_converged = self.has_converged(centers, old_centers)
            iteration_qualities.append(compute_quality_metric(data, centers))
        
        objective_function = self.compute_objective_function(distances)
        distances = cdist(data, centers)
        membership = self.compute_cluster_membership(distances)
        return membership, centers, objective_function, iteration_qualities


    def predict(self, data):
        distances = cdist(data, self.centers)
        return self.compute_cluster_membership(distances)