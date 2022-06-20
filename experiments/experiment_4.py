from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from algorithms.k_harmonic_means import KHarmonicMeans
from algorithms.fuzzy_c_means import FuzzyCMeans
from algorithms.hybrid import Hybrid1, Hybrid2
from utils.cluster_initialization import initialize_kmeans_plus_plus_centers
from utils.plot_utils import plot_k_optimization
import time
import numpy as np


def run_sklearn_kmeans(data, num_clusters, iterations, repetitions):
    kmeans = KMeans(num_clusters, n_init=repetitions, max_iter=iterations)
    clustering = kmeans.fit_predict(data)
    return clustering, kmeans.cluster_centers_


def run_fuzzy_c_means(data, num_clusters, iterations, repetitions, fuzzy_degree, convergence_threshold):
    fuzzy_c_means = FuzzyCMeans(num_clusters, iterations, fuzzy_degree, convergence_threshold, repetitions)
    centers, clustering, _ = fuzzy_c_means.fit(data, initialize_kmeans_plus_plus_centers)
    clustering = np.argmax(clustering, axis=1)
    return clustering, centers


def run_kharmonic(data, num_clusters, iterations, repetitions, p, convergence_threshold):
    kharmonic = KHarmonicMeans(num_clusters, iterations, p, convergence_threshold, repetitions)
    centers, clustering, _ = kharmonic.fit(data, initialize_kmeans_plus_plus_centers)
    clustering = np.argmax(clustering, axis=1)
    return clustering, centers


def run_hybrid_1(data, num_clusters, iterations, repetitions, p, convergence_threshold):
    kharmonic = Hybrid1(num_clusters, iterations, p, convergence_threshold, repetitions)
    centers, clustering, _ = kharmonic.fit(data, initialize_kmeans_plus_plus_centers)
    clustering = np.argmax(clustering, axis=1)
    return clustering, centers


def run_hybrid_2(data, num_clusters, iterations, repetitions, p, convergence_threshold):
    kharmonic = Hybrid2(num_clusters, iterations, p, convergence_threshold, repetitions)
    centers, clustering, _ = kharmonic.fit(data, initialize_kmeans_plus_plus_centers)
    clustering = np.argmax(clustering, axis=1)
    return clustering, centers


def run_sklearn_gaussian_EM(data, num_clusters, iterations, repetitions, convergence_threshold):
    gaussian_EM = GaussianMixture(num_clusters, n_init=repetitions, max_iter=iterations, tol=convergence_threshold)
    clustering = gaussian_EM.fit_predict(data)
    return clustering, gaussian_EM.means_


def search_optimal_number_of_clusters(data, k_values, algorithm_name, dataset_name, save_path, algorithm_function, algorithm_params):
    num_repetitions = 3
    average_silhouette_values = []
    average_calinski_values = []
    average_davis_bouldin_values = []
    print(f'Looking for the optimal number of clusters using {algorithm_name} on {dataset_name} data')

    start_time = time.time()

    for k in k_values:
        print(f'  k: {k}/{k_values[-1]}')
        sum_silhouette = 0
        sum_calinski = 0
        sum_davis = 0
        for _ in range(num_repetitions):

            clustering, _ = algorithm_function(data, k, **algorithm_params)
            sum_silhouette += silhouette_score(data, clustering)
            sum_calinski += calinski_harabasz_score(data, clustering)
            sum_davis += davies_bouldin_score(data, clustering)

        average_silhouette_values.append(sum_silhouette / num_repetitions)
        average_calinski_values.append(sum_calinski / num_repetitions)
        average_davis_bouldin_values.append(sum_davis / num_repetitions)

    results = {'Silhouette' : average_silhouette_values, 'Calinski-Harabasz' : average_calinski_values,
        'Davis&Bouldin' : average_davis_bouldin_values}

    print(f'Time to search the optimal k: {(time.time() - start_time):.2f} seconds')
    plot_k_optimization(dataset_name, k_values, results, algorithm_name, save_path)


def run_algorithm_with_optimal_k(data, optimal_k, algorithm_function, algorithm_params):
    start_time = time.time()
    clustering, centers = algorithm_function(data, optimal_k, **algorithm_params)
    end_time = time.time()
    print(f'Time to run the algorithm: {(end_time - start_time):.2f} seconds')
    return clustering, centers