from typing import Callable

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
