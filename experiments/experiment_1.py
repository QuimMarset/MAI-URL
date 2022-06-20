from algorithms.kmeans import KMeans
from algorithms.fuzzy_c_means import FuzzyCMeans
from algorithms.k_harmonic_means import KHarmonicMeans
from algorithms.hybrid import Hybrid1, Hybrid2
from algorithms.gaussian_EM import GaussianEM
from utils.metrics import *
from utils.plot_utils import *
from utils.string_utils import BIRCH
import time


def run_kmeans(data, num_clusters, initial_centers, iterations, repetitions):
    kmeans = KMeans(num_clusters, iterations, repetitions)
    centers, membership, quality_values = kmeans.fit(data, initial_centers=initial_centers)
    return centers, membership, quality_values[-1]


def run_fuzzy_c_means(data, num_clusters, initial_centers, iterations, repetitions, fuzzy_degree, convergence_threshold):
    fuzzy_c_means = FuzzyCMeans(num_clusters, iterations, fuzzy_degree, convergence_threshold, repetitions)
    centers, membership, quality_values = fuzzy_c_means.fit(data, initial_centers=initial_centers)
    return centers, membership, quality_values[-1]


def run_kharmonic(data, num_clusters, initial_centers, iterations, repetitions, p, convergence_threshold):
    kharmonic = KHarmonicMeans(num_clusters, iterations, p, convergence_threshold, repetitions)
    centers, membership, quality_values = kharmonic.fit(data, initial_centers=initial_centers)
    return centers, membership, quality_values[-1]


def run_hybrid_1(data, num_clusters, initial_centers, iterations, repetitions, p, convergence_threshold):
    kharmonic = Hybrid1(num_clusters, iterations, p, convergence_threshold, repetitions)
    centers, membership, quality_values = kharmonic.fit(data, initial_centers=initial_centers)
    return centers, membership, quality_values[-1]


def run_hybrid_2(data, num_clusters, initial_centers, iterations, repetitions, p, convergence_threshold):
    kharmonic = Hybrid2(num_clusters, iterations, p, convergence_threshold, repetitions)
    centers, membership, quality_values = kharmonic.fit(data, initial_centers=initial_centers)
    return centers, membership, quality_values[-1]


def run_gaussian_em(data, num_clusters, initial_centers, iterations, repetitions, cov_init_value, convergence_threshold):
    gaussian_em = GaussianEM(num_clusters, iterations, repetitions, convergence_threshold, cov_init_value)
    centers, membership, quality_values = gaussian_em.fit(data, initial_centers=initial_centers)
    return centers, membership, quality_values[-1]


def run_algorithm_with_birch_data(data, num_clusters, initial_centers, init_name, algorithm_name, 
                                    save_path, algorithm_function, algorithm_params):

    start_time = time.time()
    centers, membership, clustering_quality = algorithm_function(data, num_clusters, initial_centers, **algorithm_params)
    end_time = time.time()
    # In this experiment we evaluate quality using the square root of the k-means objective
    clustering_quality = np.sqrt(clustering_quality)

    clustering = np.argmax(membership, axis=1)
    plot_predicted_clustering(data, centers, clustering, BIRCH, algorithm_name, init_name, clustering_quality, save_path)

    print(f'{algorithm_name} clustering on {BIRCH} data with {init_name} initialization: quality: {clustering_quality:.2f}')
    print(f'Time to run the algorithm: {end_time - start_time:.2f}\n')