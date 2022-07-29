import os
import warnings

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from config.config_data import EXPECTED_LABEL, DATASET_LOADER, USE_RGB, MISBEHAVIOR_ONLY
from config.config_dirs import MODEL
from utils.dataset import get_train_test_data

# Ignore warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get the train and test data and labels
(train_data, train_labels), (test_data, test_labels) = get_train_test_data(
    dataset_loader=DATASET_LOADER,
    rgb=USE_RGB,
    verbose=False
)
train_data_gs, test_data_gs = (
    tf.image.rgb_to_grayscale(train_data).numpy(),
    tf.image.rgb_to_grayscale(test_data).numpy()
)
# Get the classifier
classifier = tf.keras.models.load_model(MODEL)
# Get the predictions
try:
    predictions = classifier.predict(test_data).argmax(axis=-1)
except ValueError:
    # The model expects grayscale images
    predictions = classifier.predict(test_data_gs).argmax(axis=-1)
predictions_cat = to_categorical(predictions)

# Get the mask for the data
mask_miss = np.array(test_labels != predictions)

# Filter the data if configured
if MISBEHAVIOR_ONLY:
    test_data = test_data[mask_miss]
    test_data_gs = test_data_gs[mask_miss]
    test_labels = test_labels[mask_miss]
    predictions = predictions[mask_miss]
    predictions_cat = predictions_cat[mask_miss]
