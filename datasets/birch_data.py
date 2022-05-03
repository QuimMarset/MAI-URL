import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import matplotlib.cm as cm
from itertools import cycle
import matplotlib.colors as plt_colors
from sklearn.preprocessing import StandardScaler


def generate_birch_data(grid_size=10, start_coodinate=30, points_per_cluster=100):
    xs = np.linspace(-start_coodinate, start_coodinate, grid_size)
    ys = np.linspace(-start_coodinate, start_coodinate, grid_size)
    xs, ys = np.meshgrid(xs, ys)
    cluster_centers = np.hstack((np.expand_dims(np.ravel(xs), axis=-1), np.expand_dims(np.ravel(ys), axis=-1)))

    num_clusters = grid_size * grid_size
    num_samples = num_clusters * points_per_cluster

    data, labels = make_blobs(num_samples, centers=cluster_centers, random_state=0)

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_centers = scaler.transform(cluster_centers)

    return normalized_data, labels, normalized_centers


def plot_data(data, labels, cluster_centers):
    plt.figure(figsize=(8, 8))

    num_clusters = cluster_centers.shape[0]

    # colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    color_names = list(plt_colors.cnames.keys())
    color_names.remove('black')
    colors = cycle(color_names)

    for (index, color) in zip(range(num_clusters), colors):
        cluster_data = data[labels == index]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color)

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x')
    plt.show()


if __name__ == '__main__':
    data, labels, cluster_centers = generate_birch_data()
    plot_data(data, labels, cluster_centers)