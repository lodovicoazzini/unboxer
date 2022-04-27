from clusim.clusimelement import element_sim
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from xplique.attributions import GradCAM, IntegratedGradients, Rise, KernelShap, Lime, Occlusion, \
    GuidedBackprop, GradCAMPP, SmoothGrad

from FeatureMapsClustersMode import FeatureMapsClustersMode
from utils.cluster.ClusteringMode import LocalLatentMode
from utils.cluster.postprocessing import get_frac_misses

HEATMAPS_PROCESS_MODE = LocalLatentMode
BATCH_SIZE = 64
EXPLAINERS = [
    GradCAM,
    GradCAMPP,
    GuidedBackprop,
    lambda classifier: SmoothGrad(classifier, nb_samples=50, noise=.3, batch_size=BATCH_SIZE),
    lambda classifier: IntegratedGradients(classifier, steps=50, batch_size=BATCH_SIZE),
    lambda classifier: Rise(classifier, nb_samples=4000, batch_size=BATCH_SIZE),
    lambda classifier: KernelShap(classifier, nb_samples=1000),
    lambda classifier: Lime(classifier, nb_samples=1000),
    lambda classifier: Occlusion(classifier, patch_size=10, patch_stride=10, batch_size=BATCH_SIZE),
]
DIM_RED_TECHS = [[TSNE(perplexity=perplexity)] for perplexity in range(1, 16, 2)]
CLUS_TECH = AffinityPropagation()
ITERATIONS = 20
CLUS_SIM = element_sim
FEATUREMAPS_CLUSTERS_MODE = FeatureMapsClustersMode.ORIGINAL
CLUSTERS_SORT_METRIC = lambda cluster: (
    -get_frac_misses(cluster)
    if get_frac_misses(cluster) != 1
    else 0,
    -len(cluster)
)
MAX_LABELS = 5
MAX_SAMPLES = 5
