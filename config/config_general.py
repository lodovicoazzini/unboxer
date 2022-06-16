from typing import Callable

from clusim.clusimelement import element_sim
from clusim.clustering import Clustering

from utils.clusters.extractor import get_frac_misses
from utils.images.image_similarity.geometry_based import ssim

CLUSTERS_SORT_METRIC: Callable[[list], tuple] = lambda cluster: (
    -get_frac_misses(cluster)
    if get_frac_misses(cluster) != 1
    else 0,
    -len(cluster)
)
CLUSTERS_SIMILARITY_METRIC: Callable[[Clustering, Clustering], float] = element_sim
IMAGES_SIMILARITY_METRIC: Callable = ssim
HUMAN_EVALUATION_APPROACHES = ['Lime', 'Rise', 'moves+orientation(10)_original', 'orientation+bitmaps(10)_original']
