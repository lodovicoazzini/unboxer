from typing import Callable

import numpy as np
import tensorflow as tf


def get_train_test_data(
        dataset_loader: Callable,
        rgb: bool,
        verbose: bool = False
) -> tuple:
    """
    Get the train and test data for a given dataset
    :param dataset_loader: The method to load the dataset as (train_data, train_labels), (test_data, test_labels)
    :param rgb: Whether to convert the images to rgb
    :param verbose: Whether to print some information about the imported data
    :return: (train data, train labels), (test data, test labels)
    """
    # Load the data from the dataset
    (train_data, train_labels), (test_data, test_labels) = dataset_loader()
    # Normalize the values between [0, 1]
    train_data, test_data = train_data / 255., test_data / 255.
    # Convert the images in rgb format
    if rgb:
        train_data, test_data = (
            tf.image.grayscale_to_rgb(tf.expand_dims(train_data, -1)),
            tf.image.grayscale_to_rgb(tf.expand_dims(test_data, -1))
        )

    # Print info about the data
    if verbose:
        print(f'Train samples: {train_data.shape[0]}')
        print(f'Test samples: {test_data.shape[0]}')
        print(f'Data shape: {train_data.shape[1:]}')

    return (train_data, train_labels), (test_data, test_labels)


def get_data_masks(real_labels: np.ndarray, predictions: np.ndarray, expected_label=None, verbose: bool = False):
    """
    Get the mask for the misclassified data and for the data with a label
    :param real_labels: The real labels for the data
    :param predictions: The predicted labels for the data
    :param expected_label: The expected label for which to filter the data
    :param verbose: Whether to print some information about the mask
    :return: Mask for the misclassified data, mask for the expected label
    """
    # Get the mask for the misclassified data
    misclassified_mask = real_labels != predictions
    # No expected label -> find the most misclassified label
    if expected_label is None:
        expected_label = sorted(
            list(zip(*np.unique(real_labels[misclassified_mask], return_counts=True))),
            key=lambda item: -item[1]
        )[0][0]
    # Get the mask for the expected label
    label_mask = real_labels == expected_label
    # Get the mask for the misclassified data of the expected label
    complete_mask = misclassified_mask & label_mask

    if verbose:
        num_label = len(label_mask[label_mask])
        print(f"""
Found {num_label}/{len(real_labels)} instances for the label {expected_label}
Found {len(complete_mask[complete_mask])}/{num_label} instances for misclassified {expected_label}
        """)

    return misclassified_mask, label_mask
