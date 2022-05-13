import os.path
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config.config_dirs import MERGED_DATA_SAMPLED
from steps.human_evaluation.helpers import sample_clusters
from utils import globals
from utils.cluster.postprocessing import get_misses_count, get_labels_purity
from utils.cluster.visualize import visualize_cluster_images
from utils.general import get_balanced_samples, save_figure, show_progress


def export_clusters_sample_images():
    if os.path.exists(MERGED_DATA_SAMPLED):
        df = pd.read_pickle(MERGED_DATA_SAMPLED)
    else:
        df = sample_clusters()

    # Iterate through the approaches
    df = df.set_index('approach')
    approaches = df.index.values
    with open('logs/human_evaluation_understandability_images.csv', mode='w') as file:
        for idx_approach, approach in enumerate(approaches):
            sys.stdout.write('\r')
            sys.stdout.write(
                f'{idx_approach + 1}/{len(approaches)} - {approach}\n'
            )

            # Get the clusters for the selected approach
            clusters, contributions = df.loc[approach][['clusters', 'contributions']]
            clusters = np.array(clusters, dtype=list)
            # Create the numpy array for the contributions or set it to None if no contributions
            contributions = None if type(contributions) == float else np.array(contributions, dtype=float)
            # Find the count of misclassified entries in each cluster
            counts_misses = np.vectorize(lambda cl: get_misses_count(cl, predictions=globals.predictions))(clusters)
            # Find the purity and impurity of each cluster
            purities = np.vectorize(lambda cl: get_labels_purity(cl, predictions=globals.predictions))(clusters)
            # Weight the purity and impurity based on the count of misclassified elements in log scale
            counts_misses_log = np.vectorize(lambda val: 0 if val == 0 else np.log(val))(counts_misses)
            # Get the pure and impure sample
            pure_sample, impure_sample = get_balanced_samples(clusters, sample_size=5, balanced_by=purities,
                                                              weights=counts_misses_log)
            all_samples = np.concatenate((pure_sample, impure_sample))
            show_progress(0, len(all_samples))
            for idx, cluster in enumerate(all_samples):
                fig, ax = visualize_cluster_images(
                    np.array(cluster),
                    images=globals.test_data_gs[globals.mask_label],
                    overlays=contributions,
                    predictions=globals.predictions[globals.mask_label],
                )
                plt.close(fig)

                sub_path = f'human_evaluation/understandability/{approach}_{idx}_{len(cluster)}'
                # Export the image
                save_figure(fig, f'out/{sub_path}')
                # Save teh image path in the csv file
                file.write(f'{sub_path}\n')

                show_progress(idx, len(all_samples))
            print()
