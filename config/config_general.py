from clusim.clusimelement import element_sim

from utils.cluster.postprocessing import get_frac_misses

EXPECTED_LABEL = 5
MAX_LABELS = 5
MAX_SAMPLES = 5
CLUSTERS_SORT_METRIC = lambda cluster: (
    -get_frac_misses(cluster)
    if get_frac_misses(cluster) != 1
    else 0,
    -len(cluster)
)
CLUS_SIM = element_sim

IMG_SIZE = 28
NUM_CLASSES = 10
