from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras import datasets

# DATASET_LOADER = lambda: (extract_training_samples('letters'), extract_test_samples('letters'))
DATASET_LOADER = lambda: datasets.mnist.load_data()
USE_RGB = True
EXPECTED_LABEL = 5
MISBEHAVIOR_ONLY = False
