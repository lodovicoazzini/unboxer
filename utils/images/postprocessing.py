import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker

from config.config_const import IMG_SIZE, GRID_SIZE


def add_grid(ax: plt.Axes) -> plt.Axes:
    # Set the grid intervals
    grid_interval = IMG_SIZE / GRID_SIZE
    loc = plt_ticker.MultipleLocator(base=grid_interval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Find number of sections
    nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(grid_interval)))
    ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(grid_interval)))
    # Add the labels to the grid
    for j in range(ny):
        y = grid_interval / 2 + j * grid_interval
        for i in range(nx):
            x = grid_interval / 2. + float(i) * grid_interval
            ax.text(x, y, '{:d}'.format(i + j * nx), color='k', ha='center', va='center')

    # Show the grid
    ax.grid(b=True, which='major', axis='both', linestyle='-', color='k', zorder=10)

    return ax
