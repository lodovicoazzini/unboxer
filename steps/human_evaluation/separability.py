import math
import os.path
import shutil

import numpy as np
import pandas as pd

from config.config_const import NUM_LABELABILITY_CLUSTERS
from config.config_dirs import MERGED_DATA_SAMPLED
from steps.human_evaluation.helpers import sample_clusters
from utils.clusters.extractor import get_labels_purity
from utils.clusters.postprocessing import get_misclassified_items
from utils.lists.processor import weight_values

__BASE_DIR = 'out/human_evaluation/labelability'


def export_clusters_sample_images():
    if os.path.exists(MERGED_DATA_SAMPLED):
        df = pd.read_pickle(MERGED_DATA_SAMPLED)
    else:
        df = sample_clusters()

    # Remove the data if already there
    try:
        shutil.rmtree(__BASE_DIR)
    except FileNotFoundError:
        pass

    # Iterate over the approaches
    approaches = df.index.values

    def execution(approach):
        # Get the clusters and contributions for the selected approach
        cluster_list, contributions = df.loc[approach][['clusters', 'contributions']]
        # Filter the clusters for the misclassified elements
        cluster_list = [get_misclassified_items(cluster) for cluster in cluster_list]
        # Keep the clusters with more than one misclassified element
        cluster_list = [cluster for cluster in cluster_list if len(cluster) > 1]
        cluster_list = np.array(cluster_list, dtype=list)
        # Find the clusters purity
        cluster_list_purity = [get_labels_purity(cluster) for cluster in cluster_list]
        cluster_list_purity = np.array(cluster_list_purity)
        # Find the clusters size in log scale
        cluster_list_size = [math.log(len(cluster)) for cluster in cluster_list]
        cluster_list_size = np.array(cluster_list_size)
        # Weight the purity by the log size
        cluster_list_weights = weight_values(cluster_list_purity, cluster_list_size)
        # Sort the cluster by weighted purity
        cluster_list = cluster_list[cluster_list_weights.argsort()[::-1]]
        # Select the first N clusters based on the value in the config file
        cluster_list = cluster_list[:NUM_LABELABILITY_CLUSTERS]

        # Get the central elements in the cluster
        central_elements = get_central_elements()
