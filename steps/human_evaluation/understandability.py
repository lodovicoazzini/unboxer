import shutil

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from config.config_outputs import NUM_IMAGES_PER_CLUSTER
from steps.human_evaluation.helpers import sample_clusters
from utils import global_values
from utils.clusters.extractor import get_misses_count, get_labels_purity
from utils.general import save_figure
from utils.lists.processor import get_balanced_samples
from utils.plotter.visualize import visualize_cluster_images


def export_clusters_sample_images():
    df = sample_clusters()
    # Remove the data if already there
    try:
        shutil.rmtree('out/human_evaluation/understandability')
    except FileNotFoundError:
        pass

    # Iterate through the approaches
    df = df.set_index('approach')
    approaches = df.index.values

    # Compute the titles for the images to show the predictions
    mask_label = np.array(global_values.test_labels == global_values.EXPECTED_LABEL)
    titles = [f'Predicted: {label}' for label in global_values.predictions[mask_label]]
    titles = np.array(titles)

    with open('logs/human_evaluation_understandability_images.csv', mode='w') as file:

        for approach in tqdm(approaches, desc='Exporting the clusters samples for the approaches'):
            # Get the clusters for the selected approach
            clusters, contributions = df.loc[approach][['clusters', 'contributions']]
            clusters = np.array(clusters, dtype=list)
            # Filter for the clusters with more than one element
            clusters_len = np.vectorize(len)(clusters)
            clusters = clusters[clusters_len > 1]
            # Create the numpy array for the contributions or set it to None if no contributions
            contributions = None if type(contributions) == float else np.array(contributions, dtype=float)
            # Find the count of misclassified entries in each clusters
            counts_misses = np.vectorize(lambda cl: get_misses_count(cl))(clusters)
            # Find the purity and impurity of each clusters
            purities = np.vectorize(lambda cl: get_labels_purity(cl), otypes=[np.float64])(clusters)
            # Weight the purity and impurity based on the count of misclassified elements in log scale
            counts_misses_log = np.vectorize(
                lambda val: 0 if val == 0 else np.log(val), otypes=[np.float64]
            )(counts_misses)
            # Filter out the clusters with purity = np.nan -> no misclassified entries
            clusters = clusters[~np.isnan(purities)]
            counts_misses_log = counts_misses_log[~np.isnan(purities)]
            purities = purities[~np.isnan(purities)]
            # Get the pure and impure sample
            pure_sample, impure_sample = get_balanced_samples(
                clusters,
                sample_size=5,
                balanced_by=purities,
                weights=counts_misses_log
            )
            all_samples = np.concatenate((pure_sample, impure_sample))

            for idx, sample in list(enumerate(all_samples)):
                sample = np.random.choice(sample, NUM_IMAGES_PER_CLUSTER, replace=False)
                fig, ax = visualize_cluster_images(
                    np.array(sample),
                    images=global_values.test_data_gs[mask_label],
                    overlays=contributions,
                    labels=titles
                )
                fig.suptitle(approach)
                plt.close(fig)

                num_images = min(len(sample), NUM_IMAGES_PER_CLUSTER)
                sub_path = f'human_evaluation/understandability/{approach}_{idx}_{num_images}.png'
                # Export the image
                save_figure(fig, f'out/{sub_path}')
                # Save teh image path in the csv file
                file.write(f'{sub_path}\n')
