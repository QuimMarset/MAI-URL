from algorithms.kmeans import KMeans
from algorithms.fuzzy_c_means import FuzzyCMeans
from algorithms.gaussian_EM import GaussianEM
from algorithms.k_harmonic_means import KHarmonicMeans
from algorithms.hybrid import Hybrid1, Hybrid2
from utils.metrics import *
from utils.cluster_initialization import *
from utils.plot_utils import *


def run_kmeans(data, num_clusters, initial_centers, iterations, repetitions):
    kmeans = KMeans(num_clusters, iterations, repetitions)
    _, _, quality_values = kmeans.fit(data, initial_centers=initial_centers)
    return quality_values


def run_fuzzy_c_means(data, num_clusters, initial_centers, iterations, repetitions, fuzzy_degree, convergence_threshold):
    fuzzy_c_means = FuzzyCMeans(num_clusters, iterations, fuzzy_degree, convergence_threshold, repetitions)
    _, _, quality_values = fuzzy_c_means.fit(data, initial_centers=initial_centers)
    return quality_values


def run_kharmonic(data, num_clusters, initial_centers, iterations, repetitions, p, convergence_threshold):
    kharmonic = KHarmonicMeans(num_clusters, iterations, p, convergence_threshold, repetitions)
    _, _, quality_values = kharmonic.fit(data, initial_centers=initial_centers)
    return quality_values


def run_hybrid_1(data, num_clusters, initial_centers, iterations, repetitions, p, convergence_threshold):
    kharmonic = Hybrid1(num_clusters, iterations, p, convergence_threshold, repetitions)
    _, _, quality_values = kharmonic.fit(data, initial_centers=initial_centers)
    return quality_values


def run_hybrid_2(data, num_clusters, initial_centers, iterations, repetitions, p, convergence_threshold):
    kharmonic = Hybrid2(num_clusters, iterations, p, convergence_threshold, repetitions)
    _, _, quality_values = kharmonic.fit(data, initial_centers=initial_centers)
    return quality_values


def run_gaussian_em(data, num_clusters, initial_centers, iterations, repetitions, cov_init_value, convergence_threshold):
    gaussian_em = GaussianEM(num_clusters, iterations, repetitions, convergence_threshold, cov_init_value)
    _, _, quality_values = gaussian_em.fit(data, initial_centers=initial_centers)
    return quality_values


def run_experiment_dimensions_dataset(data, num_clusters, optimal_quality_values, initial_centers, algorithm_functions, algorithms_params):
    num_algorithms = len(algorithm_functions)
    iterations = algorithms_params[0]['iterations']
    ratios = np.zeros((num_algorithms, iterations))

    for (i, (algorithm_fcn, algorithm_params)) in enumerate(zip(algorithm_functions, algorithms_params)):

        quality_values = algorithm_fcn(data, num_clusters, initial_centers, **algorithm_params)
        ratios[i] = compute_quality_metric_ratio(optimal_quality_values, quality_values, iterations)

    return ratios