import numpy as np

from config.config_featuremaps import NUM_CELLS
from feature_map.mnist.feature_simulator import FeatureSimulator
from feature_map.mnist.sample import Sample


class Feature:
    """
    Implements a feature dimension of the Feature map
    """

    def __init__(self, feature_name: str, min_value: float, max_value: float):
        self.feature_name = feature_name
        self.min_value = min_value
        self.max_value = max_value
        # get the name for the corresponding feature simulator
        self.feature_simulator = dict(FeatureSimulator.get_simulators().items())[feature_name].__name__
        self.num_cells = NUM_CELLS
        self.original_bins = np.linspace(min_value, max_value, NUM_CELLS)

    def feature_descriptor(self, sample: Sample):
        """
        Simulate the candidate solution x and record its feature descriptor
        """
        return self.feature_simulator.value(sample)

    def get_coordinate_for(self, sample: Sample):
        """
        Return the coordinate of this sample according to the definition of this axis (rescaled). It triggers exception if the
            sample does not declare a field with the name of this axis, i.e., the sample lacks this feature
        Args:
            sample:
        Returns:
            an integer representing the coordinate of the sample in this dimension in rescaled size
        Raises:
            an exception is raised if the sample does not contain the feature
        """

        # TODO Check whether the sample has the feature
        value = sample.features[self.feature_name]

        if value < self.min_value:
            print("Sample %s has value %s below the min value %s for feature %s", sample.id, value, self.min_value,
                  self.feature_name)
        elif value > self.max_value:
            print("Sample %s has value %s above the max value %s for feature %s", sample.id, value, self.max_value,
                  self.feature_name)

        return np.digitize(value, self.original_bins, right=False)

    def get_bins_labels(self):
        """
        Note that here we return explicitly the last bin
        Returns: All the bins plus the default
        """
        return self.original_bins
