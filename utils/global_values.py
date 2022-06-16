import os

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from config.config_data import EXPECTED_LABEL, DATASET_LOADER, RGB_IMAGES
from config.config_dirs import MODEL, PREDICTIONS
from steps import create_model
from utils.dataset import get_train_test_data, get_data_masks

# Prevent printing the optimization warning from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Get the train and test data and labels
(train_data, train_labels), (test_data, test_labels) = get_train_test_data(
    dataset_loader=DATASET_LOADER,
    rgb=RGB_IMAGES,
    verbose=False
)
train_data_gs, test_data_gs = (
    tf.image.rgb_to_grayscale(train_data).numpy(),
    tf.image.rgb_to_grayscale(test_data).numpy()
)
# Get the classifier
try:
    classifier = tf.keras.models.load_model(MODEL)
except IOError:
    classifier = create_model.create_model()
# Get the predictions
try:
    predictions = np.loadtxt(PREDICTIONS)
except FileNotFoundError:
    predictions = create_model.generate_predictions(classifier=classifier, test_data=test_data)
predictions_cat = to_categorical(predictions, num_classes=len(set(train_labels)))
# Get the mask for the data
mask_miss, mask_label = get_data_masks(
    real_labels=test_labels,
    predictions=predictions,
    expected_label=EXPECTED_LABEL,
    verbose=False
)
mask_miss_label = mask_miss[mask_label]
