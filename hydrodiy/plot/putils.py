import os, math, re
from datetime import datetime

from scipy.stats import gaussian_kde, chi2

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
from matplotlib.patches import Ellipse

from matplotlib import colors
from matplotlib.colors import hex2color
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.dates as mdates

import numpy as np
import pandas as pd

from scipy.stats import norm
from hydrodiy.stat import sutils, linreg

# Some useful colors
COLORS1 = '#002745'

COLORS3 = ['#FF9933', '#64A0C8', '#005BBB']

COLORS10 = [colors.rgb2hex([float(coo)/255 for coo in co]) for co in [ \
            (31, 119, 180), (255, 127, 14), (44, 160, 44), \
            (214, 39, 40), (148, 103, 189), (140, 86, 75), \
            (227, 119, 194), (127, 127, 127), (188, 189, 34), \
            (23, 190, 207)
        ] ]



def cmap2colors(ncols=10, cmap='Paired'):
    ''' generates a set of colors from colormap

    Parameters
    -----------
    ncols : int
        Number of colors
    cmap : matplotlib.colormap or str
        Colormap or colormap name

    Returns
    -----------
    colors : list
        List of colors
    '''

    cmapn = cm.get_cmap(cmap, ncols)
    return [cmapn(i) for i in range(cmapn.N)]


def colors2cmap(colors, ncols=256):
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
    >>> cmap = putils.colors2cmap(colors)
    >>> nval = 500
    >>> x = np.random.normal(size=nval)
    >>> y = np.random.normal(size=nval)
    >>> z = np.random.uniform(0, 1, size=nval)
    >>> plt.scatter(x, y, c=z, cmap=cmap)

    '''
    cdict = {
            'red': [],
            'green': [],
            'blue': []
        }

    for key in sorted(colors):
        key = float(key)
        if key < 0.:
            raise ValueError('key({0}) is lower than 0'.format(key))
        if key > 1.:
            raise ValueError('key({0}) is greater than 1'.format(key))

        col = hex2color(colors[key])

        cdict['red'].append((key, col[0], col[0]))
        cdict['green'].append((key, col[1], col[1]))
        cdict['blue'].append((key, col[2], col[2]))

    cmap = LinearSegmentedColormap('mycmap', cdict, ncols)

    return cmap


def _float(u):
    ''' Function to convert object to float '''
    try:
        v = float(u)
    except TypeError:
        v = u.toordinal()
    return v


def xdate(ax, interval='M', by=None, format='%b\n%Y'):
    ''' Format the x axis to display dates

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw the line on
    interval : str
        Interval between two tick marks. Intervals are coded as Xn
        where X is the frequency (D for days, M for months or Y for years) and n
        is the number of periods. For example 6D is 6 days.
    by : list
        Number of the month or day of the month where the ticks should be
        place. For example by=[1, 7] with internal='M' will place a tick for
        Jan and July.
    format : str
        Date format
    '''

    if interval.endswith('M'):
        if interval == 'M':
            interv = 1
        else:
            interv = int(interval[:-1])

        if by is None:
            by =range(1, 13)

        loc = mdates.MonthLocator(interval=interv, bymonth=by)

    elif interval.endswith('D'):
        if interval == 'D':
            interv = 1
        else:
            interv = int(interval[:-1])

        if by is None:
            by =[1, 10, 20]

        loc = mdates.DayLocator(interval=interv, bymonthday=by)

    elif interval.endswith('Y'):
        if interval == 'Y':
            interv = 1
        else:
            interv = int(interval[:-1])

        if by is None:
            by = [1]

        if len(by)>1:
            raise ValueError(('Expected by of length one for'+\
                ' internal=Y, got {0}').format(len(by)))

        loc = mdates.YearLocator(base=interv, month=by[0])

    else:
        raise ValueError('Expected interval to end with D, M or Y, '+\
            'got {0}'.format(interval))

    ax.xaxis.set_major_locator(loc)
    fmt = mdates.DateFormatter(format)
    ax.xaxis.set_major_formatter(fmt)



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
    preamble = mpl.rcParams['text.latex.preamble']

    # Set tex options
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=[r'\usepackage{amsmath}', \
                                    r'\usepackage{amssymb}'])

    # Plot
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

    # Restore tex options
    mpl.rc('text', usetex=usetex)
    mpl.rc('text.latex', preamble=preamble)


def set_mpl(color_theme='black', font_size=18, usetex=False):
    ''' Set convenient default matplotlib parameters

    Parameters
    -----------
    color_theme : str
        Color for text, axes and ticks
    font_size : int
        Font size
    usetex : bool
        Use tex mode or not
    '''

    # Latex mode
    mpl.rc('text', usetex=usetex)
    if usetex:
        preamble = [r'\usepackage{amsmath}', \
                        r'\usepackage{amssymb}']
        mpl.rc('text.latex', preamble=preamble)

    # Font size
    mpl.rc('font', size=font_size)

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

    # Set contour line style
    mpl.rc('contour', negative_linestyle='solid')

    # Set legend properties
    mpl.rc('legend', fancybox=True)
    mpl.rc('legend', fontsize='small')
    mpl.rc('legend', numpoints=1)
    mpl.rc('legend', markerscale=0.8)

    if not color_theme in ['k', 'black']:
        if 'legend.framealpha' in mpl.rcParams:
            mpl.rc('legend', framealpha=0.1)

    # Set colors
    mpl.rc('axes', labelcolor=color_theme)
    mpl.rc('axes', edgecolor=color_theme)
    mpl.rc('xtick', color=color_theme)
    mpl.rc('ytick', color=color_theme)
    mpl.rc('text', color=color_theme)

    if color_theme == 'white':
        if 'savefig.transparent' in mpl.rcParams:
            mpl.rc('savefig', transparent=True)


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


def cov_ellipse(mu, cov, pvalue=0.95, *args, **kwargs):
    ''' Draw ellipse contour of 2d bi-variate normal distribution

    Parameters
    -----------
    mu : numpy.ndarray
        Bi-variate mean, [2] array.
    cov : numpy.ndarray
        Bi-variate covariance matrix, [2, 2] array.
    pvalue : float
        Pvalue used to draw contour
    args, kwargs
        Argument sent to matplotlib.Patches.Ellipse
    '''
    # Check parameters
    mu = np.atleast_1d(mu)
    cov = np.atleast_2d(cov)

    if mu.shape != (2,):
        raise ValueError('Expected a [2] array for mu, got {0}'.format(\
            mu.shape))

    if cov.shape != (2,2):
        raise ValueError('Expected a [2, 2] array for cov, got {0}'.format(\
            cov.shape))

    # Compute chi square corresponding to the sum of
    # two normally distributed variables with zero means
    # unit variances
    fact = chi2.ppf(pvalue, 2)

    # Ellipse parameters
    eig, vect = np.linalg.eig(cov)
    v1 = 2*math.sqrt(fact*eig[0])
    v2 = 2*math.sqrt(fact*eig[1])
    alpha = np.rad2deg(math.acos(vect[0, 0]))

    # Draw ellipse
    ellipse = Ellipse(xy=mu, width=v1, height=v2, angle=alpha, \
            *args, **kwargs)

    return ellipse


def qqplot(ax, data, addline=False, censor=None, *args, **kwargs):
    ''' Draw a normal qq plot of data

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw the line on
    data : numpy.ndarray
        Vector data
    addline : bool
        Add the line of OLS fit
    censor : float
        Compute the OLS line above censor threshold
    '''
    datan = data[~np.isnan(data)]
    nval = len(datan)
    freqs = sutils.ppos(nval)
    xnorm = norm.ppf(freqs)
    datas = np.sort(datan)

    ax.plot(xnorm, datas, *args, **kwargs)
    ax.set_xlabel('Standard normal variable')
    ax.set_ylabel('Sorted data')

    if addline:
        idx = np.ones(nval).astype(bool)
        if not censor is None:
            idx = datas > censor + 1e-10

        lm = linreg.Linreg(xnorm[idx], datas[idx])
        lm.fit()
        a, b = lm.params['estimate']
        r2 = lm.diagnostic['R2']
        lab = 'Y = {0:0.2f} + {1:0.2f} X (r2={2:0.2f})'.format(a, b, r2)
        line(ax, 1, b, 0, a, 'k--', label=lab)

    else:
        a, b, r2 = [np.nan] * 3

    return a, b, r2


def get_fig_axs(nrows=1, ncols=1, ravel=True, close=True):
    ''' Create a figure and a set of axes.
    Ravel the set of axes if required.
    Close all figures if required.

    Parameters
    -----------
    nrows : int
        Number of rows in the set of axes
    ncols : int
        Number of columns in the set of axes
    ravel : bool
        Ravel the set of axes or not

    Returns
    -----------
    fig : matplotlib.figure.Figure
        Figure
    axs : numpy.ndarray
        Array containing matplotlib.axes.Axes. The array has dimensions
        [nrows, ncols] if ravel is True, [nrows x ncols, ] if not.
    '''

    if close:
        plt.close('all')

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    if ravel and ncols*nrows>1:
        axs = axs.ravel()

    return fig, axs
