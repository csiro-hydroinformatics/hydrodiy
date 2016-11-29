import os
import re
from datetime import datetime
import datetime

from scipy.stats import gaussian_kde

has_cycler = False
try:
    from cycler import cycler
    has_cycler = True
except ImportError:
    pass

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.path import Path

from matplotlib import colors
from matplotlib.colors import hex2color
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd

# Some useful colors
COLORS1 = '#002745'

COLORS3 = ['#FF9933', '#64A0C8', '#005BBB']

COLORS10 = [colors.rgb2hex([float(coo)/255 for coo in co]) for co in [ \
            (31, 119, 180), (255, 127, 14), (44, 160, 44), \
            (214, 39, 40), (148, 103, 189), (140, 86, 75), \
            (227, 119, 194), (127, 127, 127), (188, 189, 34), \
            (23, 190, 207)
        ] ]



def get_colors(ncols=10, palette='Paired'):
    ''' generates a set of colors '''
    cmap = cm.get_cmap(palette, ncols)
    return [cmap(i) for i in range(cmap.N)]


def col2cmap(colors):
    ''' Define a linear color map from a set of colors

    Parameters
    -----------
    colors : dict
        A set of colors indexed by a float in [0, 1]. The index
        provides the location in the color map. Example:
        colors = {'0.':'#3399FF', '0.1':'#33FFFF', '1.0':'#33FF99'}

    Returns
    -----------
    cmap : matplotlib.colormap
        Colormap

    Example
    -----------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> colors = {0.:'#3399FF', 0.1:'#33FFFF', 1.0:'#33FF99'}
    >>> cmap = putils.col2cmap(colors)
    >>> nval = 500
    >>> x = np.random.normal(size=nval)
    >>> y = np.random.normal(size=nval)
    >>> z = np.random.uniform(0, 1, size=nval)
    >>> plt.scatter(x, y, c=z, cmap=cmap)

    '''
    keys = np.sort(colors.keys()).astype(float)

    if keys[0] < 0.:
        raise ValueError('lowest key({0}) is lower than 0'.format(keys[0]))

    if keys[-1] > 1.:
        raise ValueError('Greatest key({0}) is greater than 1'.format(keys[-1]))

    cdict = {
            'red': [],
            'green': [],
            'blue': []
        }

    for k in keys:
        col = hex2color(colors[k])

        cdict['red'].append((k, col[0], col[0]))
        cdict['green'].append((k, col[1], col[1]))
        cdict['blue'].append((k, col[2], col[2]))

    return LinearSegmentedColormap('mycmap', cdict, 256)


def _float(u):
    ''' Function to convert object to float '''
    try:
        v = float(u)
    except TypeError:
        v = u.toordinal()
    return v


def line(ax, vx=0., vy=1., x0=0., y0=0., *args, **kwargs):
    ''' Plot a line following a vector (vx, vy) and
    going through the point (x0, y0). Example
    * Vertical line through (0, 0): vx=0, vy=1, x0=0, y0=0
    * Horizontal line through (0, 0): vx=1, vy=0, x0=0, y0=0
    * Line y=a+bx: vx=1, vy=a, x0=0, y0=b

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw the line on
    vx : float
        X coordinate of vector directioon
    vy : float
        Y coordinate of vector directioon
    x0 : float
        X coordinate of point
    y0 : float
        Y coordinate of point

    Returns
    -----------
    line : matplotlib.lines.Line2D
        Line drawn

    Example
    -----------
    >>> import matplotlib.pyplot as plt
    >>> from hyplot import putils
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 10], [0, 10], 'o')
    >>> putils.line(0, 1, ax, '--')
    >>> putils.line(1, 0, ax, '-', color='red')
    >>> putils.line(1, 0.5, y0=2., ax, '-', color='red')

    '''

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    vx = float(vx)
    vy = float(vy)
    if abs(vx)+abs(vy) < 1e-8:
        raise ValueError(('Both vx({0}) and vy({1}) are ' + \
            ' close to zero').format(vx, vy))

    x0 = _float(x0)
    y0 = _float(y0)

    if abs(vx)>0:
        a1 = (xlim[0]-x0)/vx
        a2 = (xlim[1]-x0)/vx
    else:
        a1 = (ylim[0]-y0)/vy
        a2 = (ylim[1]-y0)/vy

    xy0 = np.array([x0, y0])
    vxy = np.array([vx, vy])
    pt1 = xy0 + a1*vxy
    pt2 = xy0 + a2*vxy

    line = ax.plot([pt1[0], pt2[0]],
                [pt1[1], pt2[1]], *args, **kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return line


def equation(tex, filename, \
    textcolor='white', \
    transparent=True, \
    dpi = 200, \
    width = 1000, \
    height = 300):
    ''' Print latex equation into file

    Parameters
    -----------
    tex : str
        Latex equation code
    filename : str
        Filename to print in
    textcolor : str
        Text color
    transparent : bool
        Use transparent background or not
    dpi : int
        Figure resolution
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels
    fontsize : int
        Font size in points

    Example
    -----------
    >>> from hyplot import putils
    >>> tex = r'\begin{equation} s = \sum_{i=0}^{\infty} \frac{1}{i^2}
    >>> fp = '~/equation.png'
    >>> putils.equation(tex, fp)

    '''
    usetex = mpl.rcParams['text.usetex']
    mpl.rc('text', usetex=True)

    plt.close('all')

    fig, ax = plt.subplots()

    ax.text(0, 0.5, tex, color=textcolor, \
        fontsize=32, va='center')
    ax.set_ylim([0, 1.5])

    ax.axis('off')

    fig.set_size_inches(float(width)/dpi, \
                    float(height)/dpi)

    fig.tight_layout()

    fig.savefig(filename, dpi=dpi, \
        transparent=transparent)

    mpl.rc('text', usetex=usetex)


def set_spines(ax, spines='all', color='black', style='-', visible=True):
    ''' Set spines color and style '''

    if spines == 'all':
        spines = ['top', 'bottom', 'left', 'right']

    styles = { \
            ':':'dotted', \
            '-':'solid', \
            '-.':'dash_dot', \
            '--':'dashed'\
    }

    for spine in spines:
        ax.spines[spine].set_visible(visible)
        ax.spines[spine].set_color(color)

        s = style
        if style in [':', '-', '-.', '--']:
            s = styles[style]
        ax.spines[spine].set_linestyle(s)



def set_legend(leg, textcolor='black', framealpha=1):
    ''' Set legend text and transparency '''

    leg.get_frame().set_alpha(framealpha)

    for text in leg.get_texts():
        text.set_color(textcolor)


def set_mpl(reset=False):
    ''' Set convenient default matplotlib parameters

    Parameters
    -----------
    reset : bool
        Reset matplotlib config to default
    '''
    if reset:
        mpl.rcdefaults()
    else:

        # Set color cycle - depends on matplotlib version
        if 'axes.color_cycle' in mpl.rcParams:
            mpl.rc('axes', color_cycle=COLORS10)
        else:
            if has_cycler:
                mpl.rc('axes', prop_cycle=cycler('color', COLORS10))
            else:
                import warnings
                warnings.warn('Cannot set color cycle '+ \
                    'because cycler package is missing')

        # Ticker line width than default
        mpl.rc('lines', linewidth=2)

        # Set legend properties
        mpl.rc('legend', fancybox=True)
        mpl.rc('legend', fontsize='small')
        mpl.rc('legend', numpoints=1)
        mpl.rc('legend', markerscale=0.8)


def kde(xy, ngrid=50):
    ''' Interpolate a 2d pdf from a set of x/y data points using
    a Gaussian KDE. The outputs can be used to plot the pdf
    with something like matplotlib.Axes.contourf

    Parameters
    -----------
    xy : numpy.ndarray
        A set of x/y coordinates. Should be a 2d Nx2 array
    ngrid : int
        Size of grid generated

    Returns
    -----------
    xx : numpy.ndarray
        A grid ngridxngrid containing the X coordinates
    yy : numpy.ndarray
        A grid ngridxngrid containing the Y coordinates
    zz : numpy.ndarray
        A grid ngridxngrid containing the PDF estimates
    '''
    if xy.shape[1] !=2:
        xy = xy.T

    x = np.linspace(xy[:, 0].min(), xy[:, 0].max(), ngrid)
    y = np.linspace(xy[:, 1].min(), xy[:, 1].max(), ngrid)
    xx, yy = np.meshgrid(x, y)

    kd = gaussian_kde(xy.T)
    zz = kd(np.vstack([xx.ravel(), yy.ravel()]))
    zz = zz.reshape(xx.shape)

    return xx, yy, zz



