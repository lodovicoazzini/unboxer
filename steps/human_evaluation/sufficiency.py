import shutil

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from config.config_outputs import NUM_IMAGES_PER_CLUSTER
from steps.human_evaluation.helpers import sample_clusters
from utils import global_values
from utils.clusters.extractor import get_labels_purity
from utils.clusters.postprocessing import get_misclassified_items
from utils.general import save_figure
from utils.images.postprocessing import add_grid
from utils.lists.processor import get_balanced_samples
from utils.plotter.visualize import visualize_cluster_images


def export_clusters_sample_images():
    df = sample_clusters()
    # Remove the data if already there
    try:
        shutil.rmtree('out/human_evaluation/sufficiency')
    except FileNotFoundError:
        pass

    # Iterate through the approaches
    df = df.set_index('approach')
    approaches = df.index.values

    low_level_paths = []
    high_leval_paths = []
    featuremaps_appendix = '_featuremaps'

    # Compute the titles for the images to show the predictions
    titles = [f'Predicted: {label}' for label in global_values.predictions[global_values.mask_label]]
    titles = np.array(titles)

    for approach in tqdm(approaches, desc='Exporting the cluster samples for the approaches'):
        # Get the clusters for the selected approach
        clusters, contributions = df.loc[approach][['clusters', 'contributions']]
        clusters = np.array(clusters, dtype=list)
        clusters = np.array([get_misclassified_items(cluster) for cluster in clusters], dtype=list)
        # Filter for the clusters with more than one element
        clusters_len = np.vectorize(len)(clusters)
        clusters = clusters[clusters_len > 1]
        # Find the purity and impurity of each clusters
        purities = np.vectorize(lambda cl: get_labels_purity(cl), otypes=[np.float64])(clusters)
        # Get the pure and impure sample
        pure_sample, impure_sample = get_balanced_samples(
            clusters,
            sample_size=5,
            balanced_by=purities
        )
        all_samples = np.concatenate((pure_sample, impure_sample))

        for idx, sample in list(enumerate(all_samples)):
            sample = np.random.choice(sample, NUM_IMAGES_PER_CLUSTER, replace=False)
            fig, ax = visualize_cluster_images(
                np.array(sample),
                images=global_values.test_data_gs[global_values.mask_label],
                overlays=contributions,
                labels=titles
            )
            num_images = min(len(sample), NUM_IMAGES_PER_CLUSTER)

            is_featuremap = False
            if contributions is not None:
                # Low-level approach -> add the grid overlay
                for axx in ax.flatten():
                    add_grid(axx)
            else:
                is_featuremap = True
            str_appendix = featuremaps_appendix if is_featuremap else ""
            sub_path = f'human_evaluation/sufficiency{str_appendix}/{approach}_{idx}_{num_images}.png'
            plt.close(fig)
            # Export the image
            save_figure(fig, f'out/{sub_path}')
            # Add the path to the list
            if not is_featuremap:
                low_level_paths.append(sub_path)
            else:
                high_leval_paths.append(sub_path)

    # Save teh image path in the csv file
    with open(f'logs/human_evaluation_sufficiency_images.csv', mode='w') as file:
        for image_path in low_level_paths:
            file.write(f'{image_path}\n')
    with open(f'logs/human_evaluation_sufficiency_featuremaps_images.csv', mode='w') as file:
        for image_path in high_leval_paths:
            file.write(f'{image_path}\n')
