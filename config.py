from clusim.clusimelement import element_sim
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from xplique.attributions import GradCAM, SmoothGrad, IntegratedGradients, Rise, KernelShap, Lime

from FeatureMapsClustersMode import FeatureMapsClustersMode
from utils.cluster.ClusteringMode import LocalLatentMode

CLASSIFIER_PATH = 'in/models/digit_classifier.h5'
PREDICTIONS_PATH = 'in/predictions.csv'
FEATURE_MAPS_CLUSTERS_DIR = 'in/feature_map_clusters'

HEATMAPS_PROCESS_MODE = LocalLatentMode
BATCH_SIZE = 64
EXPLAINERS = [
    GradCAM,
    lambda classifier: SmoothGrad(classifier, nb_samples=50, noise=.3, batch_size=BATCH_SIZE),
    lambda classifier: IntegratedGradients(classifier, steps=50, batch_size=BATCH_SIZE),
    lambda classifier: Rise(classifier, nb_samples=4000, batch_size=BATCH_SIZE),
    lambda classifier: KernelShap(classifier, nb_samples=1000),
    lambda classifier: Lime(classifier, nb_samples=1000)
]
DIM_RED_TECHS = [TSNE(perplexity=3)]
CLUS_TECH = AffinityPropagation()
ITERATIONS = 20
CLUS_SIM = element_sim
FEATUREMAPS_CLUSTERS_MODE = FeatureMapsClustersMode.ORIGINAL
