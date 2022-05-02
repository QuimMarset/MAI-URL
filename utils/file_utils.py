import os
from constants import RESULTS_PATH, PLOTS_PATH, CLUSTERS_PATH


def create_folders():
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    os.makedirs(CLUSTERS_PATH, exist_ok=True)