from constants import *


algorithm_names = {
    KMEANS : 'k-means',
    FCMEANS : 'fuzzy_c-means',
    KHARMONIC : 'k-harmonic_means',
    GEM : 'gaussian_EM',
    HYBRID1 : 'hybrid_1',
    HYBRID2 : 'hybrid_2'
}

dataset_names = {
    BIRCH : 'birch',
    PELLEG : 'pelleg_moore'
}

initialization_names = {
    FORGY : 'forgy',
    RANDPART : 'random_partitions'
}

def get_plot_clustering_title():
    pass

def get_plot_clustering_fig_name():
    pass


def get_plot_cluster_centers_title(algorithm_name, dataset_name, initialization_name, quality, clusters_found, iterations):
    return (f'{algorithm_name} on {dataset_name} data with {initialization_name} initialization: {iterations} iterations, quality {quality:.2f},' + 
        f' clusters found {clusters_found}')


def get_plot_cluster_centers_fig_name(algorithm_name, dataset_name, initialization_name, iterations):
    return f'cluster_centers_{algorithm_names[algorithm_name]}_{dataset_names[dataset_name]}_{initialization_names[initialization_name]}_{iterations:.2f}'