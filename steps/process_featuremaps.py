import os.path
import warnings

import pandas as pd

import feature_map.mnist.feature_map as feature_map_generator
from config.config_dirs import FEATUREMAPS_DATA_RAW, FEATUREMAPS_DATA
from config.config_featuremaps import FEATUREMAPS_CLUSTERING_MODE
from utils.featuremaps.postprocessing import process_featuremaps_data

BASE_DIR = f'out/featuremaps/{FEATUREMAPS_CLUSTERING_MODE.name}'


def main():
    warnings.filterwarnings('ignore')

    featuremaps_df = feature_map_generator.main()

    # Process the feature-maps and get the dataframe
    print('Extracting the clusters data from the feature-maps ...')
    featuremaps_df = process_featuremaps_data(featuremaps_df)
    featuremaps_df.to_pickle(FEATUREMAPS_DATA)

    return featuremaps_df


if __name__ == '__main__':
    main()
