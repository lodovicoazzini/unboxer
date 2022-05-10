import sys

import pandas as pd

from config import EXPECTED_LABEL, META_FILE_DEST
from feature_simulator import FeatureSimulator
from sample import Sample
from utils import vectorization_tools
from utils.general import missing


def extract_samples_and_stats(data, labels):
    """
    Iteratively walk in the dataset and process all the json files.
    For each of them compute the statistics.
    """
    # Initialize the stats about the overall features
    stats = {feature_name: [] for feature_name in FeatureSimulator.get_simulators().keys()}

    data_samples = []
    filtered = list(filter(lambda t: t[2] == EXPECTED_LABEL, zip(range(len(data)), data, labels)))
    for idx, item in enumerate(filtered):
        seed, image, label = item
        xml_desc = vectorization_tools.vectorize(image)
        sample = Sample(seed=seed, desc=xml_desc, label=label)
        data_samples.append(sample)
        # update the stats
        for feature_name, feature_value in sample.features.items():
            stats[feature_name].append(feature_value)

        # Show the progress
        sys.stdout.write('\r')
        progress = int((idx + 1) / len(filtered) * 100)
        progress_bar_len = 20
        progress_bar_filled = int(progress / 100 * progress_bar_len)
        sys.stdout.write(f'[{progress_bar_filled * "="}{(progress_bar_len - progress_bar_filled) * " "}]\t{progress}%')
        sys.stdout.flush()
    # New line after the progress
    print()

    stats = pd.DataFrame(stats)
    # compute the stats values for each feature
    stats = stats.agg(['min', 'max', missing, 'count'])
    stats.to_csv(META_FILE_DEST, index=False)
    print(stats.transpose())

    return data_samples, stats
