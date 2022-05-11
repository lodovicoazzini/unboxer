import os.path
import warnings

import pandas as pd

import feature_map.mnist.feature_map as feature_map_generator
from config.config_dirs import FEATUREMAPS_DATA_RAW, FEATUREMAPS_DATA
from config.config_featuremaps import FEATUREMAPS_CLUSTERS_MODE
from utils.cluster.preprocessing import extract_maps_clusters

BASE_DIR = f'out/featuremaps/{FEATUREMAPS_CLUSTERS_MODE.name}'


def main():
    warnings.filterwarnings('ignore')

    # Import the featuremaps data or generate it if not there
    if os.path.exists(FEATUREMAPS_DATA_RAW):
        featuremaps_df = pd.read_pickle(FEATUREMAPS_DATA_RAW)
    else:
        featuremaps_df = feature_map_generator.main()

    # Process the feature-maps and get the dataframe
    print('Extracting the clusters data from the feature-maps ...')
    featuremaps_df = extract_maps_clusters(featuremaps_df)
    featuremaps_df.to_pickle(FEATUREMAPS_DATA)

    return featuremaps_df


if __name__ == '__main__':
    main()
