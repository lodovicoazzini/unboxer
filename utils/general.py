import os
import re
import sys
from typing import Callable, Iterable, Union

import matplotlib.pyplot as plt


def beep():
    """
    Play a sound to notify the end of an execution
    """
    os.system('say "beep"')


def save_figure(fig: plt.Figure, path: str, dpi: int = 150, transparent=True):
    """
    Save a figure to the given path, create the necessary path if not existing
    :param fig: The figure to save
    :param path: The path where to save the figure
    :param dpi: The resolution for the saved figure
    :param transparent: Whether to have a transparent background
    :return:
    """
    # Check if the path for the containing directory exists, if not -> create it
    try:
        containing_dir = re.search('.*/', path)[0]
        os.makedirs(containing_dir) if not os.path.exists(containing_dir) else None
        # Save the figure
        fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')
    except IndexError:
        raise ValueError('Invalid path')


def show_progress(
        execution: Callable,
        iterable: Iterable,
        bar_len: int = 20,
        message: Union[str, Callable[[object], str]] = None
):
    """
    Show the progress bar for the execution of a function over a list of entries
    :param execution: The execution to run on every entry
    :param iterable: The list of entries o which to run the execution function
    :param bar_len: The length of the progress bar to show
    :param message: The message to show for each iteration, can be a function on the entry
    :return: The final return value of the execution function
    """
    # Keep track of the result
    result = None
    # Avoid deleting the last printed line
    print() if message is not None else None
    # Echo the initial progress bar
    for idx, value in enumerate(iterable):
        # Execute the code on the current value
        try:
            result = execution(*value)
        except TypeError:
            result = execution(value)
        # Echo the progress bar
        progress = int(idx / (len(list(iterable)) - 1) * 100)
        progress_bar_filled = int(progress / 100 * bar_len)
        progress_str = f'[{progress_bar_filled * "="}{(bar_len - progress_bar_filled) * " "}]\t{progress}%'
        if message is not None:
            try:
                message_str = message(value)
            except TypeError:
                message_str = message
            echo_str = f'{progress_str} ({idx}/{len(list(iterable)) - 1}):\t{message_str}\r'
        else:
            echo_str = f'{progress_str}\r'
        sys.stdout.write(echo_str)
        # sys.stdout.flush()
    # New line at the end of the execution
    print()
    # return the result of the execution
    return result
