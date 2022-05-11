from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from xplique.attributions import DeconvNet, Occlusion, Saliency, GuidedBackprop, Lime, GradCAM, IntegratedGradients, \
    KernelShap, SmoothGrad, GradCAMPP, Rise

from utils.cluster.ClusteringMode import LocalLatentMode

__batch_size = 64

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
ITERATIONS = 20
