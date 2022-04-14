from FeatureMapsClustersMode import FeatureMapsClustersMode
from utils.cluster.ClusteringMode import LocalLatentMode

CLASSIFIER_PATH = 'in/models/digit_classifier.h5'
PREDICTIONS_PATH = 'in/predictions.csv'
FEATURE_MAPS_CLUSTERS_DIR = 'in/feature_map_clusters'

HEATMAPS_PROCESS_MODE = LocalLatentMode
FEATUREMAPS_CLUSTERS_MODE = FeatureMapsClustersMode.ORIGINAL
