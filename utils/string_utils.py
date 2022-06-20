
# Constants storing the names of the algorithms, datasets, and initialization methods

# Algorithm names
KMEANS = 'K-Means'
FCMEANS = 'Fuzzy C-Means'
KHARMONIC = 'K-Harmonic Means'
GEM = 'Gaussian EM'
HYBRID1 = 'Hybrid 1'
HYBRID2 = 'Hybrid 2'
SKLEARN_KMEANS = 'Sklearn K-Means'
SKLEARN_GEM = 'Sklearn Gaussian EM'

# Dataset names
BIRCH = 'BIRCH'
PELLEG = 'Pelleg and Moore'
ADULT = 'adult'

# Initialization method names
FORGY = 'Forgy'
RANDPART = 'Random partitions'
KMEANSPLUS = 'K-Means++'



# Dictionaries storing the conversion of the names to use them in file names

algorithm_names_in_files = {
    KMEANS : 'k-means',
    FCMEANS : 'fuzzy_c-means',
    KHARMONIC : 'k-harmonic_means',
    GEM : 'gaussian_EM',
    HYBRID1 : 'hybrid_1',
    HYBRID2 : 'hybrid_2',
    SKLEARN_KMEANS : 'sklearn_kmeans',
    SKLEARN_GEM : 'sklearn_gaussian_em',
    'Ground-truth' : 'ground-truth'
}

dataset_names_in_files = {
    BIRCH : 'birch',
    PELLEG : 'pelleg_moore',
    ADULT : 'adult'
}

initialization_names_in_files = {
    FORGY : 'forgy',
    RANDPART : 'random_partitions',
    KMEANSPLUS : 'kmeans++'
}