import os, math, re
import warnings
from datetime import datetime

from scipy.stats import gaussian_kde, chi2, norm

HAS_CYCLER = False
try:
    from cycler import cycler
    HAS_CYCLER = True
except ImportError:
    pass

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.path import Path
from matplotlib.patches import Ellipse

from matplotlib import colors as mcolors
from matplotlib.colors import hex2color, rgb2hex
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.dates as mdates

import numpy as np
import pandas as pd

from scipy.stats import norm
from hydrodiy.stat import sutils, linreg

# Some useful colors
COLORS_SLIDE_BACKGROUND = '#002745'
COLORS_BADGOOD = ['#4D935F', '#915592']
COLORS_TERCILES = ['#FF9933', '#64A0C8', '#005BBB']
COLORS_TAB = [mcolors.rgb2hex([float(coo)/255 for coo in co]) for co in [ \
            (31, 119, 180), (255, 127, 14), (44, 160, 44), \
            (214, 39, 40), (148, 103, 189), (140, 86, 75), \
            (227, 119, 194), (127, 127, 127), (188, 189, 34), \
            (23, 190, 207)
        ] ]

# Palette for color blind readers
# see https://www.somersault1824.com/tips-for-designing-scientific-figures-for-color-blind-readers/
COLORS_CBLIND = ['#000000', '#074751', '#009292', '#FE6CB5', '#FEB5DA', \
    '#490092', '#006DDB', '#B66CFE', '#6DB6FE', '#B6DBFF', \
    '#920000', '#924900', '#DB6D00', '#23FE22', '#FFFF6D']


# Palette safe for a range of uses (cf PuOr palette)
# see http://colorbrewer2.org/
COLORS_SAFE = ['#E66101', '#FDB863', '#B2ABD2', '#5E3C99']


def cmap2colors(ncols=10, cmap='Paired'):
    ''' Generates a set of colors from a colormap

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

    if isinstance(cmap, str):
        if cmap == 'safe':
            cmapn = colors2cmap({0.:COLORS_SAFE[0], 0.5: '#A0A0A0', 1.:COLORS_SAFE[-1]})
        else:
            cmapn = cm.get_cmap(cmap, ncols)

        return [rgb2hex(cmapn(i)) for i in range(cmapn.N)]
    else:
        ii = np.linspace(0, cmap.N, ncols+2)
        ii = np.round(ii).astype(int)[1:-1]
        return [rgb2hex(cmap(i)) for i in ii]


def colors2cmap(colors, ncols=256):
    ''' Define a linear color map from a set of colors

    Parameters
    -----------
    colors : dict
        A set of colors indexed by a float in [0, 1]. The index
        provides the location in the color map. Example:
        colors = {0.:'#3399FF', 0.1:'#33FFFF', 1.0:'#33FF99'}

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


def cmap2grayscale(cmap):
    ''' Return a grayscale version of the given colormap

    This code was pasted from
    https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html

    Parameters
    -----------
    cmap : matplotlib.colormap
        Colormap object

    Returns
    -----------
    grayscale : matplotlib.colormap
        Colormap object containing gray scale

    '''
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    grayscale = LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

    return grayscale


def cmap_accentuate(cmap, param=-1, ninterp=100):
    ''' Accentuate a cmap contrast

    Parameters
    -----------
    cmap : matplotlib.colors.Cmap
        Input color palette
    param : float
        Parameter controlling the intensity of the
        accentuation.
        * -inf : no accentuation
        * -1 : nearly no accentuation
        * +inf : maximum accentuation
    ninterp : int
        Number of interpolation point for the output cmap

    Returns
    -----------
    mcmap : matplotlib.colors.Cmap
        Accentuated color palette
    '''
    if param < -1 or param > 4:
        raise ValueError('Expected param in [-1, 4], got {0}'.format(param))

    # Accentuating function (mapping from [0, 1] to [0, 1])
    def fun(x):
        a = math.exp(param)
        v0 = math.tanh(-0.5*a)
        v1 = math.tanh(0.5*a)
        return (np.tanh((x-0.5)*a)-v0)/(v1-v0)

    # Get original cmap colors
    colors = cmap2colors(256, cmap)

    # Build dict
    ws = np.linspace(0, 1, ninterp)
    mdict = {w: colors[int(round(fun(w)*255))] for w in ws}

    # Create a linearly interpolated cmap
    mcmap = colors2cmap(mdict)

    return mcmap


def cmap_neutral(cmap, band_width=0.05, \
            neutral_color='#C0C0C0', ninterp=100):
    ''' Replace the central part of a color map with a
        neutral color

    Parameters
    -----------
    cmap : matplotlib.colors.Cmap
        Input color palette
    band_width : float
        Width of the neutral band
    neutral_color : str
        Neutral color
    ninterp : int
        Number of interpolation point for the output cmap

    Returns
    -----------
    mcmap : matplotlib.colors.Cmap
        Accentuated color palette
    '''
    # Get original cmap colors
    colors = cmap2colors(256, cmap)

    # Build dict
    ws = np.linspace(0, 1, ninterp)
    mdict = {w: colors[int(round(w*255))] for w in ws}

    # Replace with neutral color
    idx = np.abs(ws-0.5) < band_width
    for w in ws[idx]:
        mdict[w] = neutral_color

    # Create a linearly interpolated cmap
    mcmap = colors2cmap(mdict)

    return mcmap


def interpolate_color(color, amount=0., between=['k', 'w']):
    ''' Interpolate color linearly between two extremes.

    Parameters
    -----------
    color : str or tuple
        Color name or rgb
    amount : float
        Interpolation index
        * 0 : color = between[0]
        * 1 : color = between[1]
    between : list
        Two colors to define the interpolation extremes.
        Default is black to white.

    Returns
    -----------
    color : str
        HGB color

    Examples:
    -----------
    >> interpolate_color('g', 0.3)
    >> interpolate_color('#F034A3', 0.6)
    >> interpolate_color((.3,.55,.1), 0.5)
    '''
    # Check inputs
    amount = float(amount)
    if amount < 0. or amount > 1.:
        raise  ValueError('Expected amount in [0, 1], got {0}'.format(\
                        amount))

    if len(between) != 2:
        raise  ValueError('Expected len(between)=2, got {0}'.format(\
                        len(between)))

    # Build interpolated color map
    cmap = colors2cmap({0.:between[0], 0.5:color, 1:between[1]})

    # Return interpolated color
    i = int(round(cmap.N * amount))
    return rgb2hex(cmap(i))


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
        where X is the frequency (D for days, M for months or Y for years)
        and n is the number of periods. For example 6D is 6 days.
    by : list
        Number of the month or day of the month where the ticks should be
        place. For example by=[1, 7] with internal='M' will place a tick for
        Jan and July.
    format : str
        Date format
    '''

    # Check ax x data seems reasonable
    xticks = ax.get_xticks()
    if np.any(xticks > 1e7):
        raise ValueError('xaxis does not seem to contain python datetime '+
            'values. If plotting pandas data, use the to_pydatetime() '+
            'function')

    # Compute locator parameter
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
        Latex equation code including begin and end statements
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


def set_mpl(color_theme='black', font_size=18, usetex=False, \
                    color_cycle=COLORS_TAB):
    ''' Set convenient default matplotlib parameters

    Parameters
    -----------
    color_theme : str
        Color for text, axes and ticks
    font_size : int
        Font size
    usetex : bool
        Use tex mode or not
    color_cycle : list
        List of colors to cycle through.
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
        mpl.rc('axes', color_cycle=color_cycle)
    else:
        if HAS_CYCLER:
            mpl.rc('axes', prop_cycle=cycler('color', color_cycle))
        else:
            warnings.warn('Cannot set color cycle '+ \
                'because cycler package is missing')

    # Default colormap
    mpl.rc('image', cmap='PiYG')

    # Ticker line width than default
    mpl.rc('lines', linewidth=2)

    # Set contour line style
    mpl.rc('contour', negative_linestyle='solid')

    # Set legend properties
    mpl.rc('legend', fancybox=True)
    mpl.rc('legend', fontsize='small')
    mpl.rc('legend', labelspacing=0.8)
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


def kde(xy, ngrid=50, eps=1e-10):
    ''' Interpolate a 2d pdf from a set of x/y data points using
    a Gaussian KDE. The outputs can be used to plot the pdf
    with something like matplotlib.Axes.contourf

    Parameters
    -----------
    xy : numpy.ndarray
        A set of x/y coordinates. Should be a 2d Nx2 array
    ngrid : int
        Size of grid generated
    eps : float
        Random error added to avoid singular matrix error
        when x or y have ties.

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

    if eps > 0.:
        xy += np.random.uniform(-eps, eps, size=xy.shape)

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
    alpha = np.sign(cov[0, 1])*np.rad2deg(math.acos(vect[0, 0]))

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


def ecdfplot(ax, df, label_stat=None, label_stat_format='4.2f', \
            cst=0., *args, **kwargs):
    ''' Plot empirical cumulative density functions

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw the line on
    df : pandas.core.dataframe.DataFrame
        Input data
    label_stat : str
        Statistic use for the label, should be an attribute
        of pandas.Series (e.g. mean, median or nunique).
        If None, does not print the label stat.
    label_stat_format : str
        Format to use for the label statistic value.
    cst : float
        Constant used to compute plotting positions
        See hydrodiy.stat.sutils.ppos
    args, kwargs
        Argument sent to matplotlib.pyplot.plot command for each

    Returns
    -----------
    lines : dict
        Dictionnary containing the line object for each column in df.
    '''
    lines = {}
    for name, se in df.iteritems():
        values = se.sort_values()
        values = values[~np.isnan(values)]

        pp = sutils.ppos(len(values), cst=cst)

        label = name
        if not label_stat is None:
            stat = getattr(se, label_stat)()
            label = '{} ({:{format}})'.format(name, stat, format=label_stat_format)

        ax.plot(values, pp, label=label, *args, **kwargs)
        lines[name] = ax.get_lines()[-1]

    # Decorate
    ax.set_ylabel('Empirical CDF [-]')
    ax.set_yticks([0., 0.5, 1.])
    ax.set_yticklabels(['0', chr(189)+' ', '1'])
    ylabs = ax.get_yticklabels()
    ylabs[0].set_va('bottom')
    ylabs[2].set_va('top')
    ax.set_ylim((0, 1))

    return lines


def scattercat(ax, x, y, z, ncats=5, cuts=None, cmap='viridis', \
                        fmt='0.2f', nval=False, *args, **kwargs):
    ''' Draw a scatter plot using different colors depending on categories
    defined by z. Be careful when z has a lot of zeros, quantile computation
    may lead to non-unique category boundaries.

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw the points on
    x : numpy.ndarray
        X coordinates
    y : numpy.ndarray
        Y coordinates
    z : numpy.ndarray
        Z values to derive categories
    ncats : int
        Number of categories when creating categories from quantiles.
        This value is ignored if cuts is not None
    cuts : list
        Bounds to create categories from z values.
    cmap : str
        Matplotlib color map name to change color for each category.
        Input data
    fmnt : str
        Number format to be used in labels
    nval : bool
        Add number of points in labels
    args, kwargs
        Argument sent to matplotlib.pyplot.plot command

    Returns
    -----------
    plotted : list
        List of dictionaries for each categories. A dictionary contains:
        idx :   Index of category items
        label:  Label of category
        color:  Color used for category
        line:   matplotlib.Line object
        x:      X coordinate
        y:      Y coordinate

    cats : pandas.Series
        Series containing the category number for each item
    '''
    z = pd.Series(z)

    # Check z is categorical
    # See  https://pandas.pydata.org/pandas-docs/version/0.17.0/categorical.html
    # Cell [176]
    if hasattr(z, 'cat'):
        # Use categorical data properties
        z = pd.Categorical(z)
        labels = z.categories.values
        ncats = len(labels)
        cats = z.codes
    else:
        if ncats is not None or cuts is not None:
            if cuts is None:
                # Create categories
                qq = np.linspace(0, 1, ncats+1)
                cuts = list(z.quantile(qq))

            # make sure the cuts cover the full range
            cuts[0] = z.min()-1e-10
            cuts[-1] = z.max()+1e-10

            # Create categories
            ncats = len(cuts)-1
            cats = pd.cut(z, cuts, right=True, labels=False).astype(int)

            if len(set(cuts)) != len(cuts):
                raise ValueError(\
                        'Non-unique category boundaries :{0}'.format(\
                            '/ '.join([str(u) for u in list(cuts)])))

            # Create labels
            labels = ['[{0:{fmt}}, {1:{fmt}}]'.format(cuts[icat], \
                                           cuts[icat+1], fmt=fmt) \
                                               for icat in range(ncats)]
        else:
            raise ValueError('Expected ncats or cuts to be not-None')

    # Get colors for each category
    colors = cmap2colors(ncats, cmap)

    # Plot all categories
    plotted = []

    for icat in range(ncats):
        # Plot category
        idx = cats == icat
        label = labels[icat]

        if np.sum(idx) > 0:
            u, v = x[idx], y[idx]
        else:
            u, v = [], []
            warnings.warn('No points falling in category '+\
                        '{0} ({1})'.format(icat, label))

        ax.plot(u, v, 'o', color=colors[icat], label=label, \
                                *args, **kwargs)

        line = ax.get_lines()[-1]

        # Store plotted data
        dd = {'idx': idx, \
            'label': label, \
            'color': colors[icat], \
            'line': line, \
            'x': x[idx], 'y': y[idx]}

        plotted.append(dd)

    return plotted, cats


def bivarnplot(ax, xy, add_semicorr=True, namex='var 1', \
                namey='var 2', marker='o', *args, **kwargs):
    ''' Bivariate normal scores Plot

    Useful to check symetry of correlation.
    Semi-correlation are added in the top left corner.
    See hydrodiy.stat.sutils.semicorr

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw the line on
    xy : numpy.ndarray
    add_semicorr : bool
        Add the semi correlation sample and theoretical values
    namex : str
        Name for variable displayed in X axis
    namey : str
        Name for variable displayed in Y axis
    marker : str
        Marker to use for the points displayed
    args, kwargs
        Argument sent to matplotlib.pyplot.plot command for each
    '''
    # Select data
    idx = np.sum(np.isnan(xy), axis=1) == 0
    if np.sum(idx) < 2:
        raise ValueError('Expected at least 2 data pairs with valid'+\
                ' values for both, got {}'.format(np.sum(idx)))
    xy = xy[idx]

    # Compute normal standard variables and semi correlations
    unorm = np.zeros_like(xy)
    unorm[:, 0], _ = sutils.standard_normal(xy[:, 0])
    unorm[:, 1], _ = sutils.standard_normal(xy[:, 1])
    rho, eta, rho_p, rho_m = sutils.semicorr(unorm)

    # Plot
    ax.plot(unorm[:, 0], unorm[:, 1], marker, *args, **kwargs)

    line(ax, 1, 0, 0, 0, 'k--', lw=0.6)
    line(ax, 0, 1, 0, 0, 'k--', lw=0.6)

    # Add semi correlations
    if add_semicorr:
        text = r'$\rho$   {:5.2f}'.format(rho)
        text += '\n'+r'$\eta$   {:5.2f}'.format(eta)
        text += '\n'+r'$\rho^+$ {:5.2f}'.format(rho_p)
        text += '\n'+r'$\rho^-$ {:5.2f}'.format(rho_m)
        ax.text(0.02, 0.98, text, transform=ax.transAxes, \
                va='top', ha='left')
    # Decorate
    ax.set_xlabel('Standard normal score for {} [-]'.format(namex))
    ax.set_ylabel('Standard normal score for {} [-]'.format(namey))



