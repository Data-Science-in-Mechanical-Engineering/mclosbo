import os
import glob
import json
import matplotlib.colors as mc
import colorsys
from PIL import Image


def get_environment_path(venv_name):
    """Get the absolute path of the virtual environment.

    Parameters
    ----------
    venv_name: str
        Name of the current virtual environment

    Returns
    -------
    str
        Absolute path to the current virtual environment
    """
    path = os.getcwd()
    start_idx = path.rfind(venv_name)
    path = path[:start_idx + len(venv_name) + 1]
    if not path[-1] == "/":
        path += "/"
    return path


def removeImages(img_path):
    for f in glob.glob(img_path):
        os.remove(f)


def createGIF(img_path, save_path, delete_imgs=False, duration=200):
    """Create a GIF from images. Use name formating as `name = f'/test_gif_{id:03d}.png'`.

    Parameters
    ----------
    img_path: str
        Path to the images as shown in the example.
    save_path: str
        Path and name of the GIF.
    delete_imgs: bool, default=False
        Whether to delete the images or not ate creating the GIF.
    duration: int, default=200
        duration per frame in milliseconds

    Returns
    -------

    Examples
    -------
    >>> path = get_environment_path(venv_name="visualizations-with-matplotlib")
    >>> createGIF(path + "/images/test_gif_*", path + "/test_gif.gif", delete_imgs=True, duration=800)

    """
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    all_imgs = sorted(glob.glob(img_path))
    img, *imgs = [Image.open(f) for f in all_imgs]
    img.save(fp=save_path, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=0)
    # if delete images after creating the GIF
    if delete_imgs:
        removeImages(img_path)


def load_colors(venv_name):
    """Get the official RWTH Aachen University colors.

    Parameters
    ----------
    venv_name: str
        Name of the current virtual environment

    Returns
    -------
    dict
        Dictionary with all the RWTH Aachen University colors.
    """
    path = get_environment_path(venv_name)
    color_json = glob.glob(path + '/**/RWTHcolors.json', recursive=True)
    with open(color_json[0]) as json_file:
        c = json.load(json_file)
    return c


def initialize_plot(conference):
    """Get parameters for matplotlib.

    Parameters
    ----------
    conference: str
        Name of the conference for default values. (So far: {'AAAI', 'CDC'})

    Returns
    -------
    dict:
        Dictionary with the parameters

    Examples
    -------
    >>> import matplotlib as mpl
    >>> params = initialize_plot('AAAI')
    >>> mpl.rcParams.update(params)
    """

    if conference == 'AAAI':
        plot_params = {
            "font.family": "serif",
            "text.usetex": True,
            'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}'
        }
    elif conference == 'CDC':
        plot_params = {
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': True,
            'axes.labelsize': 9,
            'ytick.labelsize': 9,
            'xtick.labelsize': 9,
            "pgf.preamble": "\n".join([
                 r'\usepackage{bm}',
            ]),
            #changed
            'text.latex.preamble': "\n".join([r'\usepackage{amsmath}',
                                r'\usepackage{amssymb}',
                                r'\usepackage{bm}'])
        }

    elif conference == 'README':
        plot_params = {
            'font.family': 'serif',
            'text.usetex': True,
            'axes.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.labelsize': 10,
            "legend.fontsize": 10,
            "pgf.preamble": "\n".join([
                 r'\usepackage{bm}',
            ]),
            'text.latex.preamble': [r'\usepackage{amsmath}',
                                    r'\usepackage{amssymb}',
                                    r'\usepackage{bm}'],
        }
    else:
        plot_params = {}

    return plot_params


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to sit nicely in our document.
    Use `\the\textwidth` in your latex document to get the valid width in pts.

    Parameters
    ----------
    width_pt: float
        Document width in points
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
        The number of rows and columns of subplots.

    Returns
    -------
    tuple
        Dimensions of figure in inches

    Examples
    -------
    >>> fig_x, fig_y = set_size(398, subplots=(1, 2))
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def lighten_color(color, amount=0.5):
    """Lighten a given color. Note that the result will no longer be compliant
    with the RWTH Aachen University color formatting.

    Parameters
    ----------
    color
        Color as matplotlib color string, hex string, or RGB tuple
    amount: float, default=0.5
        Amount by which the given color is multiplied (1-luminosity).

    Returns
    -------
    tuple
        RGB tuple with the lighter color.

    Examples
    --------
    >>> lighten_color('g', 0.3)
    >>> lighten_color('#F034A3', 0.6)
    >>> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
