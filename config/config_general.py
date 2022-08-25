from typing import Callable

import numpy as np
from clusim.clusimelement import element_sim
from utils.clusters.similarity_metrics import intra_pairs_similarity
from clusim.clustering import Clustering

from utils.clusters.extractor import get_frac_misses
from utils.images.image_similarity.geometry_based import ssim
from utils.images.image_similarity.intensity_based import euclidean_similarity, mean_squared_similarity
from utils.images.postprocessing import mask_noise

CLUSTERS_SORT_METRIC: Callable[[list], tuple] = lambda cluster: (
    -get_frac_misses(cluster)
    if get_frac_misses(cluster) != 1
    else 0,
    -len(cluster)
)
CLUSTERS_SIMILARITY_METRIC: Callable[[Clustering, Clustering], float] = element_sim


def IMAGES_SIMILARITY_METRIC(lhs, rhs, threshold: float = None, max_activation: float = None, num_bins: int = 2):
    # lhs_processed = lhs
    # rhs_processed = rhs
    # if threshold is not None:
    #     lhs_processed, _ = mask_noise(lhs_processed, normalize=True, threshold=threshold)
    #     rhs_processed, _ = mask_noise(rhs_processed, normalize=True, threshold=threshold)
    # if max_activation is not None:
    #     lhs_processed = np.digitize(lhs_processed, np.linspace(0, max_activation, num_bins))
    #     rhs_processed = np.digitize(rhs_processed, np.linspace(0, max_activation, num_bins))
    return mean_squared_similarity(lhs, rhs)


HUMAN_EVALUATION_APPROACHES = [
    'Lime',
    'Rise',
    'moves+orientation(10)_original',
    'orientation+bitmaps(10)_original'
]
