import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from algorithms.base_clustering import BaseClustering
from utils.metrics import compute_quality_metric
from sklearn.cluster import KMeans


class GaussianEM(BaseClustering):

    def __init__(self, num_clusters, max_iterations, repetitions, threshold, cov_init_value):
        super().__init__(num_clusters, max_iterations)
        self.threshold = threshold
        self.cov_init_value = cov_init_value
        self.repetitions = repetitions


    def init_gaussian_parameters(self, data, initial_mus):
        num_dimensions = data.shape[1]
        # Regularization factor to avoid singular covariance matrices
        self.reg_cov = 1e-6*np.identity(num_dimensions)

        # Mus for each gaussian
        self.mus = initial_mus

        # Symmetric covariance matrices
        self.cov_matrices = np.zeros((self.num_clusters, num_dimensions, num_dimensions))
        for index in range(self.num_clusters):
            np.fill_diagonal(self.cov_matrices[index], self.cov_init_value)

        # Cluster prior (i.e. probability of a random point to belong to a cluster)
        self.pis = np.ones(self.num_clusters) / self.num_clusters


    def perform_expectation_step(self, data):
        gaussians = [multivariate_normal(mu, cov_matrix) for (mu, cov_matrix) in zip(self.mus, self.cov_matrices+self.reg_cov)]
        log_pdfs = np.array([gaussian.logpdf(data) for gaussian in gaussians])
        numerator = log_pdfs + np.log(self.pis)[:, np.newaxis]
        denominator = logsumexp(numerator, axis=0)
        return np.mean(denominator), numerator - denominator[np.newaxis, :]


    def perform_maxmization_step(self, data, log_posterior_data_cluster):
        posterior_data_cluster = np.exp(log_posterior_data_cluster)
        weights = self.compute_data_weights(data)
        self.mus = self.recompute_center_location(data, posterior_data_cluster.T, weights)
        prob_belonging = np.sum(posterior_data_cluster, axis=1)

        diff = data - np.expand_dims(self.mus, axis=1)
        self.cov_matrices = (1 / prob_belonging)[:, np.newaxis, np.newaxis] * \
            np.matmul(np.transpose(posterior_data_cluster[:, :, np.newaxis] * diff, axes=[0, 2, 1]), diff) + self.reg_cov

        self.pis = prob_belonging / data.shape[0]
        

    def compute_cluster_membership(self, data):
        _, membership = self.perform_expectation_step(data)
        return membership.T


    def compute_data_weights(self, data):
        return np.ones(data.shape[0])


    def compute_objective_function(self, data):
        gaussians = [multivariate_normal(mu, cov_matrix) for (mu, cov_matrix) in zip(self.mus, self.cov_matrices)]
        pdfs = np.array([gaussian.pdf(data) for gaussian in gaussians])
        log_likelihood = -np.sum(np.log(np.sum([pi * pdf for (pi, pdf) in zip(self.pis, pdfs)], axis=0)))
        return log_likelihood


    def has_converged(self, log_likelihood):
        change = abs(log_likelihood - self.prev_log_likelihood)
        return change < self.threshold


    def fit_repetition(self, data, initial_centers):
        self.init_gaussian_parameters(data, initial_centers)
        iteration = 0
        has_converged = False
        log_likelihood = np.inf
        # Used in the second experiment
        iteration_qualities = []

        while iteration < self.max_iterations and not has_converged:
            self.prev_log_likelihood = log_likelihood

            log_likelihood, log_posterior_data_cluster = self.perform_expectation_step(data)
            self.perform_maxmization_step(data, log_posterior_data_cluster)
            
            iteration += 1
            has_converged = self.has_converged(log_likelihood)
            iteration_qualities.append(compute_quality_metric(data, self.mus))
        
        membership = self.compute_cluster_membership(data)
        return membership, self.mus, log_likelihood, iteration_qualities


    def predict(self, data):
        return self.compute_cluster_membership(data)