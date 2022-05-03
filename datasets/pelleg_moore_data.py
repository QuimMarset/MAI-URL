from gym import make
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import matplotlib.cm as cm
from itertools import cycle
import matplotlib.colors as plt_colors
from sklearn.preprocessing import StandardScaler


def sample_unit_hypercube(dimensions, num_samples):
    samples = np.zeros((num_samples, dimensions))
    for i in range(dimensions):
        samples[:, i] = np.random.uniform(-0.5, 0.5, num_samples)
    return samples


def generate_pelleg_moore_data(dimensions=2, num_clusters=50, samples_per_cluster=50, cluster_std_factor=0.012):
    cluster_centers = sample_unit_hypercube(dimensions, num_clusters)
    num_samples = num_clusters * samples_per_cluster
    data, labels = make_blobs(num_samples, n_features=dimensions, centers=cluster_centers, cluster_std=dimensions*cluster_std_factor)

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_centers = scaler.transform(cluster_centers)

    return normalized_data, labels, normalized_centers


def plot_data(data, labels, cluster_centers):
    plt.figure(figsize=(8, 8))
    num_dimensions = data.shape[1]

    num_clusters = cluster_centers.shape[0]

    color_names = list(plt_colors.cnames.keys())
    color_names.remove('black')
    colors = cycle(color_names)

    if num_dimensions < 3:
        for (index, color) in zip(range(num_clusters), colors):
            cluster_data = data[labels == index]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x')

    else:
        ax = plt.axes(projection='3d')
        for (index, color) in zip(range(num_clusters), colors):
            cluster_data = data[labels == index]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], color=color)
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], color='black', marker='x')

    plt.show()


if __name__ == '__main__':
    data, labels, cluster_centers = generate_pelleg_moore_data(dimensions=3)
    plot_data(data, labels, cluster_centers)
