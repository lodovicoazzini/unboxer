import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from config.config_dirs import MODEL, PREDICTIONS
from config.config_general import EXPECTED_LABEL
from utils.dataset import get_train_test_data, get_data_masks

print('This should happen only once')

# Get the train and test data and labels
(train_data, train_labels), (test_data, test_labels) = get_train_test_data(rgb=True, verbose=True)
train_data_gs, test_data_gs = (
    tf.image.rgb_to_grayscale(train_data).numpy(),
    tf.image.rgb_to_grayscale(test_data).numpy()
)
# Get the classifier
try:
    classifier = tf.keras.models.load_model(MODEL)
except IOError:
    classifier = model.create_model()
# Get the predictions
try:
    predictions = np.loadtxt(PREDICTIONS)
except FileNotFoundError:
    predictions = model.generate_predictions(classifier=classifier, test_data=test_data)
predictions_cat = to_categorical(predictions, num_classes=len(set(train_labels)))
# Get the mask for the data
mask_miss, mask_label = get_data_masks(real=test_labels, predictions=predictions, label=EXPECTED_LABEL, verbose=True)
mask_miss_label = mask_miss[mask_label]
