import pandas as pd

from config.config_data import EXPECTED_LABEL
from config.config_dirs import FEATUREMAPS_META
from feature_map.mnist.feature_simulator import FeatureSimulator
from feature_map.mnist.sample import Sample
from feature_map.mnist.utils import vectorization_tools
from feature_map.mnist.utils.general import missing
from utils.general import show_progress


def extract_samples_and_stats(data, labels):
    """
    Iteratively walk in the dataset and process all the json files.
    For each of them compute the statistics.
    """
    print('Extracting the samples and the statistics ...')
    # Initialize the stats about the overall features
    stats = {feature_name: [] for feature_name in FeatureSimulator.get_simulators().keys()}

    data_samples = []
    filtered = list(filter(lambda t: t[2] == EXPECTED_LABEL, zip(range(len(data)), data, labels)))

    def execution(seed, image, label):
        xml_desc = vectorization_tools.vectorize(image)
        sample = Sample(seed=seed, desc=xml_desc, label=label)
        data_samples.append(sample)
        # update the stats
        for feature_name, feature_value in sample.features.items():
            stats[feature_name].append(feature_value)

    show_progress(execution=execution, iterable=filtered)

    stats = pd.DataFrame(stats)
    # compute the stats values for each feature
    stats = stats.agg(['min', 'max', missing, 'count'])
    print(stats.transpose())
    stats.to_csv(FEATUREMAPS_META, index=True)

    return data_samples, stats
