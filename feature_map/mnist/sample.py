import json
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from config.config_data import EXPECTED_LABEL
from feature_map.mnist.feature_simulator import FeatureSimulator
from feature_map.mnist.utils import rasterization_tools
from utils.global_values import predictions


class Sample:
    def __init__(self, desc, label, prediction):
        self.id = id(self)
        self.xml_desc = desc
        self.expected_label = label
        self.predicted_label = prediction
        # if self.predicted_label != EXPECTED_LABEL:
        #     print(f'predicted label: {self.predicted_label}, real label: {label}, idx: {self.seed}')
        # assert self.predicted_label != EXPECTED_LABEL
        self.features = {
            feature_name: feature_simulator(self)
            for feature_name, feature_simulator in FeatureSimulator.get_simulators().items()
        }

    def to_dict(self):
        return {'id': id(self),
                'expected_label': self.expected_label,
                'predicted_label': self.predicted_label,
                'misbehaviour': self.is_misbehavior,
                'features': self.features
                }

    @property
    def purified_image(self):
        return rasterization_tools.rasterize_in_memory(self.xml_desc)

    @property
    def is_misbehavior(self):
        return self.expected_label != self.predicted_label

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
