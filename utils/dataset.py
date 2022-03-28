import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.python.module import module


def get_train_test_data(
        dataset: module = datasets.mnist,
        normalize: bool = True,
        rgb: bool = True,
        verbose: bool = False):
    # load the train and test data for the MNIST dataset
    (train_data, train_labels), (test_data, test_labels) = dataset.load_data()

    # normalize the values between [0, 1]
    if normalize:
        train_data, test_data = train_data / 255., test_data / 255.

    # convert the images in rgb format
    if rgb:
        train_data, test_data = (
            tf.image.grayscale_to_rgb(tf.expand_dims(train_data, -1)),
            tf.image.grayscale_to_rgb(tf.expand_dims(test_data, -1))
        )

    # print info about the data
    if verbose:
        print(f'Train samples: {train_data.shape[0]}')
        print(f'Test samples: {test_data.shape[0]}')
        print(f'Data shape: {train_data.shape[1:]}')

    return (train_data, train_labels), (test_data, test_labels)


def get_data_mask(real: np.ndarray, predictions: np.ndarray, label=None, verbose: bool = False):
    # get the mask for the misclassified data
    misclassified_mask = real != predictions
    # find the most misclassified label if not provided
    if label is None:
        label = sorted(
            list(zip(*np.unique(real[misclassified_mask], return_counts=True))),
            key=lambda item: -item[1]
        )[0][0]
    # get the mask for the misclassified_label
    label_mask = real == label
    # get the complete mask
    complete_mask = misclassified_mask & label_mask

    if verbose:
        print(f"""
Selected {len(complete_mask[complete_mask == True])}/{len(real)} instances for misclassified {label}
        """)

    return complete_mask
