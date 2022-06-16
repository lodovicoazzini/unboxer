from typing import Callable

from utils.featuremaps.FeaturemapsClusteringMode import FeaturemapsClusteringMode
from utils.images.image_similarity.geometry_based import ssim

NUM_CELLS: int = 10
BITMAP_THRESHOLD: float = 0.5
ORIENTATION_THRESHOLD: float = 0.
FEATUREMAPS_CLUSTERING_MODE: FeaturemapsClusteringMode = FeaturemapsClusteringMode.ORIGINAL
IMAGES_DISTANCE_METRIC: Callable = ssim
