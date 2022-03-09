import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.python.module import module


def get_train_test_data(
        dataset: module = datasets.mnist,
        normalize: bool = True,
        rgb: bool = True,
        verbose: bool = False):
    """
    Get the train and test data from one tensorflow.keras.datasets.
    :param dataset: The dataset to use [should be from tensorflow.keras.datasets or provide a method load_data()]
    :param normalize: Normalize the images between [0, 1]
    :param rgb: Make grayscale images to rgb
    :param verbose: Print some information about the loaded data
    :return: (train_data, train_labels), (test_data, test_labels)
    """
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
