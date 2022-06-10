from typing import Callable

from clusim.clusimelement import element_sim
from clusim.clustering import Clustering

from utils.clusters.extractor import get_frac_misses
from utils.image_similarity.intensity_based import euclidean_distance

CLUSTERS_SORT_METRIC: Callable[[list], tuple] = lambda cluster: (
    -get_frac_misses(cluster)
    if get_frac_misses(cluster) != 1
    else 0,
    -len(cluster)
)
CLUSTERS_SIMILARITY_METRIC: Callable[[Clustering, Clustering], float] = element_sim
HEATMAPS_SIMILARITY_METRIC: Callable = euclidean_distance
