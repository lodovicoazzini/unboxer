import os

import matplotlib.pyplot as plt


def beep():
    os.system('say "beep"')


def save_figure(fig: plt.Figure, path: str, dpi: int = 150, transparent=True):
    # check if the containing directory exists
    out_dir = '/'.join(path.split('/')[:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save the figure
    fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')
