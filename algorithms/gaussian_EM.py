import numpy as np
from scipy.stats import multivariate_normal
from algorithms.base_clustering import BaseClustering


class GaussianEM(BaseClustering):

    def __init__(self, num_clusters, max_iterations, threshold, cov_init_value):
        super().__init__(num_clusters, max_iterations)
        self.threshold = threshold
        self.cov_init_value = cov_init_value


    def init_gaussian_parameters(self, data, initial_mus):
        num_dimensions = data.shape[1]
        # Regularization factor to avoid singular covariance matrices
        self.reg_cov = 1e-6*np.identity(num_dimensions)

        # Mus for each gaussian
        if initial_mus is None:
            self.mus = self.initialize_cluster_centers(data)
        else:
            self.mus = initial_mus

        # Symmetric covariance matrices
        self.cov_matrices = np.zeros((self.num_clusters, num_dimensions, num_dimensions))
        for index in self.num_clusters:
            np.fill_diagonal(self.cov_matrices[index], self.cov_init_value)

        # Cluster prior (i.e. probability of a random point to belong to a cluster)
        self.pis = np.ones(self.num_clusters) / self.number_of_sources


    def perform_expectation_step(self, data):
        num_points = data.shape[0]
        posterior_data_cluster = np.zeros((self.num_clusters, num_points))

        gaussians = [multivariate_normal(mu, cov_matrix) for (mu, cov_matrix) in zip(self.mus, self.cov_matrices+self.reg_cov)]
        pdfs = np.array([gaussian.pdf(data) for gaussian in gaussians])

        for cluster_index, pi, in zip(range(self.num_clusters), self.pis):
            pdf = pdfs[cluster_index]
            numerator = pi * pdf
            denominator = np.sum(self.pis * pdfs, axis=0)
            posterior_data_cluster[cluster_index, :] = numerator / denominator

        return posterior_data_cluster


    def perform_maxmization_step(self, data, posterior_data_cluster):
        fraction_points = np.sum(posterior_data_cluster, axis=0)
        self.pis = fraction_points / np.sum(posterior_data_cluster)
        weights = self.compute_data_weights(data)
        self.mus = self.recompute_center_location(data, posterior_data_cluster, weights)
        self.cov_matrices = np.dot(posterior_data_cluster * (data - self.mus).T, (data - self.mus)) / fraction_points


    def compute_cluster_membership(self, data):
        return self.perform_expectation_step(data)


    def compute_data_weights(self, data):
        return np.ones(data.shape[0])


    def compute_objective_function(self, data):
        gaussians = [multivariate_normal(mu, cov_matrix) for (mu, cov_matrix) in zip(self.mus, self.cov_matrices)]
        pdfs = np.array([gaussian.pdf(data) for gaussian in gaussians])
        log_likelihood = -np.sum(np.log(np.sum([pi * pdf for (pi, pdf) in zip(self.pis, pdfs)], axis=0)))
        return log_likelihood


    def has_converged(self, log_likelihood):
        change = abs(log_likelihood - self.prev_log_likelihood)
        self.prev_log_likelihood = log_likelihood
        return change < self.threshold


    def fit(self, data, initial_centers):
        self.prev_log_likelihood = np.inf
        self.init_gaussian_parameters(data, initial_centers)
        iteration = 0
        has_converged = False

        while iteration < self.max_iterations and not has_converged:
            posterior_data_cluster = self.perform_expectation_step(data)
            self.perform_maxmization_step(data, posterior_data_cluster)
            
            iteration += 1
            log_likelihood = self.compute_objective_function(self, data)
            has_converged = self.has_converged(log_likelihood)

        membership = self.compute_cluster_membership(data)
        return membership, self.mus, log_likelihood


    def predict(self, data):
        return self.compute_cluster_membership(data)