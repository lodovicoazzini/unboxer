import json
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from feature_map.mnist import predictor
from feature_map.mnist.feature_simulator import FeatureSimulator
from feature_map.mnist.utils import rasterization_tools


class Sample:
    def __init__(self, seed, desc, label):
        self.id = id(self)
        self.seed = seed
        self.xml_desc = desc
        self.expected_label = label
        self.predicted_label, self.confidence = predictor.Predictor.predict(self.purified_image)
        self.features = {
            feature_name: feature_simulator(self)
            for feature_name, feature_simulator in FeatureSimulator.get_simulators().items()
        }

    def to_dict(self):
        return {'id': id(self),
                'seed': self.seed,
                'expected_label': self.expected_label,
                'predicted_label': self.predicted_label,
                'misbehaviour': self.is_misbehavior,
                'performance': self.confidence,
                'features': self.features
                }

    @property
    def purified_image(self):
        return rasterization_tools.rasterize_in_memory(self.xml_desc)

    @property
    def is_misbehavior(self):
        return self.expected_label != self.predicted_label

    def evaluate(self):
        """
        Compute the fitness function
        """
        # Calculate fitness function
        return self.confidence if self.confidence > 0 else -0.1

    def from_dict(self, the_dict):
        for k in self.__dict__.keys():
            if k in the_dict.keys():
                setattr(self, k, the_dict[k])
        return self

    def dump(self, filename):
        data = self.to_dict()
        filedest = filename + ".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

    def save_png(self, filename):
        plt.imsave(filename + '.png', self.purified_image.reshape(28, 28), cmap='gray', format='png')

    def save_npy(self, filename):
        np.save(filename, self.purified_image)
        test_img = np.load(filename + '.npy')
        diff = self.purified_image - test_img
        assert (np.linalg.norm(diff) == 0)

    def save_svg(self, filename):
        data = self.xml_desc
        filedest = filename + ".svg"
        with open(filedest, 'w') as f:
            f.write(data)

    def export(self, dst):
        dst = join(dst, "mbr" + str(self.id))
        self.dump(dst)
        self.save_npy(dst)
        self.save_png(dst)
        self.save_svg(dst)

    def clone(self):
        clone_digit = Sample(seed=self.seed, desc=self.xml_desc, label=self.expected_label)
        return clone_digit
