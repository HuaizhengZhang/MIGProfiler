import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf

mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

WIDTH = 506.295
MARKER_LIST = ['.', '*', '^', 'p', 'o', 'x']
HATCH_LIST = [None, '\\', 'x', '-', '/', '+']
COLOR_LIST = ['#DC143C', '#8B008B', '#6495ED', '#3CB371', '#FFD700', '#F08080', '#FF8C00', '#008B8B', '#7B68EE']


def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 1.5

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)
    #     fig_dim = (2*fig_width_in, fig_height_in)

    return fig_dim


def set_style():
    plt.style.use('classic')

    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 19,
        "font.size": 16,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }

    mpl.rcParams.update(nice_fonts)


def load_json(path):
    '''
    Load a JSON local file as a dict()
    :param path: input path
    :return: JSON dict()
    '''
    with open(path) as f:
        data = json.load(f)
    return data
