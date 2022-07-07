import math
import os.path
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from config.config_general import HUMAN_EVALUATION_APPROACHES, IMAGES_SIMILARITY_METRIC
from config.config_outputs import NUM_IMAGES_PER_CLUSTER, NUM_SEPARABILITY_CLUSTERS
from steps.human_evaluation.helpers import sample_clusters
from utils import global_values
from utils.clusters.extractor import get_labels_purity, get_central_elements
from utils.clusters.postprocessing import get_misclassified_items
from utils.general import save_figure
from utils.lists.processor import weight_values
from utils.plotter.visualize import visualize_cluster_images

__BASE_DIR = 'out/human_evaluation/separability'


def export_clusters_sample_images():
    df = sample_clusters()
    # Remove the data if already there
    try:
        shutil.rmtree(__BASE_DIR)
    except FileNotFoundError:
        pass

    # Iterate over the approaches
    df = df.set_index('approach')
    approaches_df = []
    for approach in HUMAN_EVALUATION_APPROACHES:
        try:
            approaches_df.append(df.loc[[approach]])
        except KeyError:
            continue
    if len(approaches_df) < len(HUMAN_EVALUATION_APPROACHES):
        # Sample to reach the same length
        remaining = list(set(df.index) - set(HUMAN_EVALUATION_APPROACHES))
        for approach_df in [
            df.loc[[approach]]
            for approach
            in np.random.choice(
                remaining,
                min(len(HUMAN_EVALUATION_APPROACHES) - len(approaches_df), len(remaining)),
                replace=False
            )
        ]:
            approaches_df.append(approach_df)
    df = pd.concat(approaches_df, axis=0)
    approaches = df.index.values

    for approach in tqdm(approaches, desc='Collecting the data for the approaches'):
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
        cluster_list = cluster_list[:NUM_SEPARABILITY_CLUSTERS]
        # Get the contributions or the images themselves
        label_images = global_values.test_data_gs[global_values.mask_label]

        # Process the clusters
        for idx, cluster in tqdm(
                list(enumerate(cluster_list)),
                desc='Visualizing the central elements fo the clusters',
                leave=False
        ):
            # Get the central elements in the cluster
            central_elements = get_central_elements(
                cluster,
                cluster_elements=contributions[cluster] if contributions is not None else label_images[cluster],
                elements_count=NUM_IMAGES_PER_CLUSTER,
                metric=lambda lhs, rhs: 1 - IMAGES_SIMILARITY_METRIC(lhs, rhs),
                show_progress_bar=False
            )
            central_elements = np.array(central_elements)
            # Visualize the central elements
            fig, ax = visualize_cluster_images(
                central_elements,
                images=label_images,
                labels='auto'
            )
            plt.close(fig)
            save_figure(fig, os.path.join(__BASE_DIR, f'{approach}_{idx}'))

        # # Combine the images for the approach
        # approach_path = os.path.join(__BASE_DIR, approach)
        # images_paths = [os.path.join(approach_path, img_path) for img_path in os.listdir(approach_path)]
        # # Get all the combinations of images
        # images_paths_combinations = list(combinations(images_paths, 2))
        #
        # for idx, t in enumerate(images_paths_combinations):
        #     lhs_path, rhs_path = t
        #     fig, ax = combine_images(lhs_path, rhs_path)
        #     ax[0].set_title('First cluster'), ax[1].set_title('Second cluster')
        #     plt.tight_layout()
        #     save_figure(fig, os.path.join(__BASE_DIR, f'{approach}_{idx}'))
