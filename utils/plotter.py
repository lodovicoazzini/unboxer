import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

fontsize_title = 18

_positive_contributions = lambda contributions: [
    (
        np.ma.masked_less(np.squeeze(contributions), 0).filled(0),
        'Greens_r'
    )
]
_negative_contributions = lambda contributions: [
    (
        np.ma.masked_greater(np.squeeze(contributions), 0).filled(0),
        'Reds'
    )
]
_absolute_contributions = lambda contributions: [
    (
        np.apply_along_axis(lambda val: abs(val), -1, np.squeeze(contributions)),
        'Greens_r'
    )
]


class ContributionsMode(Enum):
    NONE = lambda contributions: (None, None, None)
    POSITIVE = lambda contributions: (
        _positive_contributions(contributions),
        'Positive'
    )
    NEGATIVE = lambda contributions: (
        _negative_contributions(contributions),
        'Negative',
    )
    ABSOLUTE = lambda contributions: (
        _absolute_contributions(contributions),
        'Absolute',
    )
    ALL = lambda contributions: (
        _positive_contributions(contributions) + _negative_contributions(contributions),
        'Positive and Negative',
    )


def show_image(image, real, predicted, cmap='gray', **args):
    """
    Show an image with its label (and prediction).
    :param cmap: The colormap to use
    :param image: The image to show
    :param real: The label of the image
    :param predicted: The predicted label for the image (leave empty if no prediction)
    :param args: The args for plt.subplots()
    :return: figure and axes
    """
    # figure settings
    fig, ax = plt.subplots(**args)
    title = f'Real: {real}'
    if predicted:
        title += f'\nPredicted: {predicted}'
    ax.set_title(title, fontsize=fontsize_title)
    # style
    ax.axis('off')

    # show the image
    ax.imshow(image, cmap=cmap)
    plt.show()

    return fig, ax


def show_random_image(images, labels, **args):
    """
    Show a random image from a dataset.
    :param images: The images in the dataset
    :param labels: The labels in the dataset
    :param args: The args for plt.subplots()
    :return:
    """
    # make sure that images and labels are same length
    assert images.shape[0] == labels.shape[0]

    # get the index for a random image
    idx = random.randint(0, images.shape[0])

    # show the random image
    return show_image(images[idx], labels[idx], **args)


def show_contributions(
        original=None, real=None, predicted=None,
        contributions=None, mode=ContributionsMode.NONE,
        cmap=None,
        **args):
    """
    Show the contributions for the explanations of an image prediction.
    :param original: The original image
    :param real: The ground truth label for the image
    :param predicted: The predicted label for the image
    :param contributions: The contributions for the image explanation
    :param mode: How to show the contributions (see ContributionsMode)
    :param cmap: The colormap to use for the contributions (default greens for positive and reds for negative)
    :param args: The arguments for plt.subplots
    :return: figure and axes
    """
    # create the overall image
    fig, ax = plt.subplots(**args)
    # style
    ax.axis('off')
    title_builder = []

    # plot the original image if provided
    if original is not None:
        ax.imshow(original, cmap='gray')
        title_builder.append(f'Real: {real}') if real is not None else None
        title_builder.append(f'Predicted: {predicted}') if predicted is not None else None

    # if the contribution mode is set to NONE or no contributions are provided exit
    if mode is ContributionsMode.NONE or contributions is None:
        return fig, ax

    # convert to grayscale
    if len(contributions.shape) > 3:
        contributions = np.mean(contributions, -1)

    # show the contributions based on the type
    layers, contributions_name = mode(contributions)
    # set the title
    title_builder.append(f'{contributions_name} contributions')
    ax.set_title('\n'.join(title_builder), fontsize=fontsize_title)
    # for each layer, show the contributions
    for layer in layers:
        layer_contributions, layer_cmap = layer
        # remove the 0 values
        layer_contributions[layer_contributions == 0] = np.nan
        layer_img = ax.imshow(layer_contributions, cmap=layer_cmap if cmap is None else cmap, alpha=.8)
        plt.colorbar(layer_img)

    return fig, ax
