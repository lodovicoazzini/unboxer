from clusim.clusimelement import element_sim
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from xplique.attributions import GradCAM, IntegratedGradients, Rise, KernelShap, Lime, Occlusion, \
    GuidedBackprop, GradCAMPP, SmoothGrad, DeconvNet, Saliency

from utils.cluster.ClusteringMode import LocalLatentMode
from utils.cluster.FeatureMapsClustersMode import FeatureMapsClustersMode
from utils.cluster.postprocessing import get_frac_misses

__batch_size = 64

CHOSEN_LABEL = 5
ITERATIONS = 20
MAX_LABELS = 5
MAX_SAMPLES = 5

HEATMAPS_PROCESS_MODE = LocalLatentMode
EXPLAINERS = [
    DeconvNet,
    lambda classifier: Occlusion(classifier, patch_size=10, patch_stride=10, batch_size=__batch_size),
    Saliency,
    GuidedBackprop,
    lambda classifier: Lime(classifier, nb_samples=100),
    GradCAM,
    lambda classifier: IntegratedGradients(classifier, steps=50, batch_size=__batch_size),
    lambda classifier: KernelShap(classifier, nb_samples=100),
    lambda classifier: SmoothGrad(classifier, nb_samples=100, noise=.3, batch_size=__batch_size),
    GradCAMPP,
    lambda classifier: Rise(classifier, nb_samples=4000, batch_size=__batch_size)
]
DIM_RED_TECHS = [
    [TSNE(perplexity=perplexity)]
    for perplexity in range(1, 20, 2)
]
CLUS_TECH = AffinityPropagation()
CLUS_SIM = element_sim

FEATUREMAPS_CLUSTERS_MODE = FeatureMapsClustersMode.ORIGINAL

CLUSTERS_SORT_METRIC = lambda cluster: (
    -get_frac_misses(cluster)
    if get_frac_misses(cluster) != 1
    else 0,
    -len(cluster)
)
