import os, math, re
import warnings
from datetime import datetime
from getpass import getuser
from scipy.stats import gaussian_kde, chi2, norm
from scipy import linalg

import colorsys

from PIL import Image
from PIL.PngImagePlugin import PngImageFile, PngInfo

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
from hydrodiy.stat import sutils

# Some useful colors
COLORS_BADGOOD = ["#4D935F", "#915592"]
COLORS_TERCILES = ["#FF9933", "#64A0C8", "#005BBB"]

# Palette for color blind readers
# see https://www.somersault1824.com/tips-for-designing-scientific-figures-for-color-blind-readers/
COLORS_CBLIND = ["#000000", "#074751", "#009292", "#FE6CB5", "#FEB5DA", \
    "#490092", "#006DDB", "#B66CFE", "#6DB6FE", "#B6DBFF", \
    "#920000", "#924900", "#DB6D00", "#23FE22", "#FFFF6D"]


# Palette safe for a range of uses (cf PuOr palette)
# see http://colorbrewer2.org/
COLORS_SAFE = ["#E66101", "#FDB863", "#B2ABD2", "#5E3C99"]

# Useful palette
COLORS_CORE = {
    "middayblue": "#00a9ce", \
    "midnightblue": "#001d34", \
    "steel": "#757579", \
    "mist": "#dadbdc"
}

COLORS_PRIMARY = {
    "blueberry": "#1e2277", \
    "oceanblue": "#004b87", \
    "teal": "#007377", \
    "mint": "#007a53"
}

COLORS_SECONDARY = {
    "plum": "#6d2077", \
    "fuschia": "#df1995", \
    "orange": "#e77722", \
    "gold": "#e1b81c", \
    "lavender": "#9faee5", \
    "lightteal": "#2dccd3", \
    "forest": "#78be20", \
    "lightmint": "#71cc98"
}

def _float(u):
    """ Function to convert object to float """
    try:
        v = float(u)
    except TypeError:
        v = u.toordinal()
    return v



def blackwhite(fimg, prefix="BW_"):
    fimgbw = fimg.parent / f"{prefix}{fimg.name}"
    im = Image.open(fimg).convert("L")
    im.save(fimgbw)


def darken_or_lighten(colname, modifier):
    """ Generates lighter (factor<0) or darker (factor>0) color.
    Based on https://stackoverflow.com/questions/37765197/
                            darken-or-lighten-a-color-in-matplotlib
    Parameters
    -----------
    colname : str
        Color name or HEX code.
    Modifier : float
        Color modifier.

    Returns
    -----------
    colors : list
        List of colors
    """
    if colname == "none":
        return colname

    try:
        colcode = mcolors.cnames[colname]
    except:
        colcode = colname

    colhls = colorsys.rgb_to_hls(*mcolors.to_rgb(colcode))
    # Get color luminosity
    lum = colhls[1]
    assert lum<1, f"Expected hls lum<1 for color {colname}"

    vmax = 1/(1-lum)
    shift = math.atanh(1-2*lum)
    assert abs(modifier)<2, "Expected |modifier|<2"
    m = vmax*(1+np.tanh(modifier+shift))/2

    # Change luminosity
    modif_lum = 1-m*(1-lum)
    return colorsys.hls_to_rgb(colhls[0], modif_lum, colhls[2])


def cmap2colors(ncols=10, cmap="Paired"):
    """ Generates a set of colors from a colormap

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
    """

    if isinstance(cmap, str):
        if cmap == "safe":
            cmapn = colors2cmap({0.:COLORS_SAFE[0], 0.5: "#A0A0A0", 1.:COLORS_SAFE[-1]})
        else:
            cmapn = cm.get_cmap(cmap, ncols)

        return [rgb2hex(cmapn(i)) for i in range(cmapn.N)]
    else:
        ii = np.linspace(0, cmap.N, ncols+2)
        ii = np.round(ii).astype(int)[1:-1]
        return [rgb2hex(cmap(i)) for i in ii]


def colors2cmap(colors, ncols=256):
    """ Define a linear color map from a set of colors

    Parameters
    -----------
    colors : dict
        A set of colors indexed by a float in [0, 1]. The index
        provides the location in the color map. Example:
        colors = {0.:"#3399FF", 0.1:"#33FFFF", 1.0:"#33FF99"}

    Returns
    -----------
    cmap : matplotlib.colormap
        Colormap

    Example
    -----------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> colors = {0.:"#3399FF", 0.1:"#33FFFF", 1.0:"#33FF99"}
    >>> cmap = putils.colors2cmap(colors)
    >>> nval = 500
    >>> x = np.random.normal(size=nval)
    >>> y = np.random.normal(size=nval)
    >>> z = np.random.uniform(0, 1, size=nval)
    >>> plt.scatter(x, y, c=z, cmap=cmap)

    """
    cdict = {
            "red": [],
            "green": [],
            "blue": []
        }

    for key in sorted(colors):
        key = float(key)
        if key < 0.:
            raise ValueError("key({0}) is lower than 0".format(key))
        if key > 1.:
            raise ValueError("key({0}) is greater than 1".format(key))

        col = hex2color(colors[key])

        cdict["red"].append((key, col[0], col[0]))
        cdict["green"].append((key, col[1], col[1]))
        cdict["blue"].append((key, col[2], col[2]))

    cmap = LinearSegmentedColormap("mycmap", cdict, ncols)

    return cmap



def line(ax, vx=0., vy=1., x0=0., y0=0., *args, **kwargs):
    """ Plot a line following a vector (vx, vy) and
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
    >>> ax.plot([0, 10], [0, 10], "o")
    >>> putils.line(0, 1, ax, "--")
    >>> putils.line(1, 0, ax, "-", color="red")
    >>> putils.line(1, 0.5, y0=2., ax, "-", color="red")

    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    vx = float(vx)
    vy = float(vy)
    if abs(vx)+abs(vy) < 1e-8:
        raise ValueError(("Both vx({0}) and vy({1}) are " + \
            " close to zero").format(vx, vy))

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


def set_mpl(color_theme="black", font_size=18, usetex=False, \
                    color_cycle=mcolors.TABLEAU_COLORS):
    """ Set convenient default matplotlib parameters

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
    """

    # Latex mode
    mpl.rc("text", usetex=usetex)
    if usetex:
        preamble = r"\usepackage{amsmath}\n"+ \
                        r"\usepackage{amssymb}"
        mpl.rc("text.latex", preamble=preamble)

    # Font size
    mpl.rc("font", size=font_size)

    # Set color cycle - depends on matplotlib version
    if "axes.color_cycle" in mpl.rcParams:
        mpl.rc("axes", color_cycle=color_cycle)
    else:
        if HAS_CYCLER:
            mpl.rc("axes", prop_cycle=cycler("color", color_cycle))
        else:
            warnings.warn("Cannot set color cycle "+ \
                "because cycler package is missing")

    # Default colormap
    mpl.rc("image", cmap="PiYG")

    # Ticker line width than default
    mpl.rc("lines", linewidth=2)

    # Set contour line style
    mpl.rc("contour", negative_linestyle="solid")

    # Set legend properties
    mpl.rc("legend", fancybox=True)
    mpl.rc("legend", fontsize="small")
    mpl.rc("legend", labelspacing=0.8)
    mpl.rc("legend", numpoints=1)
    mpl.rc("legend", markerscale=0.8)

    if not color_theme in ["k", "black"]:
        if "legend.framealpha" in mpl.rcParams:
            mpl.rc("legend", framealpha=0.1)

    # Set colors
    mpl.rc("axes", labelcolor=color_theme)
    mpl.rc("axes", edgecolor=color_theme)
    mpl.rc("xtick", color=color_theme)
    mpl.rc("ytick", color=color_theme)
    mpl.rc("text", color=color_theme)

    if color_theme == "white":
        if "savefig.transparent" in mpl.rcParams:
            mpl.rc("savefig", transparent=True)


def kde(xy, ngrid=50, eps=1e-10):
    """ Interpolate a 2d pdf from a set of x/y data points using
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
    """
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
    """ Draw ellipse contour of 2d bi-variate normal distribution

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
    """
    # Check parameters
    mu = np.atleast_1d(mu)
    cov = np.atleast_2d(cov)

    if mu.shape != (2,):
        raise ValueError("Expected a [2] array for mu, got {0}".format(\
            mu.shape))

    if cov.shape != (2,2):
        raise ValueError("Expected a [2, 2] array for cov, got {0}".format(\
            cov.shape))

    # Compute chi square corresponding to the sum of
    # two normally distributed variables with zero means
    # unit variances
    fact = chi2.ppf(pvalue, 2)

    # Ellipse parameters
    eig, vect = linalg.eig(cov)
    eig = eig.real
    v1 = 2*math.sqrt(max(1e-6, fact*eig[0]))
    v2 = 2*math.sqrt(max(1e-6, fact*eig[1]))
    alpha = np.sign(cov[0, 1])*np.rad2deg(math.acos(vect[0, 0]))

    # Draw ellipse
    ellipse = Ellipse(xy=mu, width=v1, height=v2, angle=alpha, \
            *args, **kwargs)

    return ellipse


def qqplot(ax, data, addline=False, censor=None, *args, **kwargs):
    """ Draw a normal qq plot of data

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
    """
    datan = data[~np.isnan(data)]
    nval = len(datan)
    freqs = sutils.ppos(nval)
    xnorm = norm.ppf(freqs)
    datas = np.sort(datan)

    ax.plot(xnorm, datas, *args, **kwargs)
    ax.set_xlabel("Standard normal variable")
    ax.set_ylabel("Sorted data")

    if addline:
        idx = np.ones(nval).astype(bool)
        if not censor is None:
            idx = datas > censor + 1e-10

        x, y = xnorm[idx], datas[idx]

        # Fit OLS regression
        X = np.column_stack([np.ones(len(x)), x])
        theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b = theta

        # Compute r2 for regression
        yy = np.column_stack([y, X.dot(theta)])
        r2 = np.corrcoef(yy.T)[0, 1]

        lab = f"Y = {a:0.2f} + {b:0.2f} X (r2={r2:0.2f})"
        line(ax, 1, b, 0, a, "k--", label=lab)
    else:
        a, b, r2 = [np.nan] * 3

    return a, b, r2


def ecdfplot(ax, df, label_stat=None, label_stat_format="4.2f", \
            cst=0., *args, **kwargs):
    """ Plot empirical cumulative density functions

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
    """
    lines = {}
    for name, se in df.iteritems():
        values = se.sort_values().values
        values = values[~np.isnan(values)]

        pp = sutils.ppos(len(values), cst=cst)

        label = name
        if not label_stat is None:
            stat = getattr(se, label_stat)()
            label = "{} ({:{format}})".format(name, stat, \
                                format=label_stat_format)

        ax.plot(values, pp, label=label, *args, **kwargs)
        lines[name] = ax.get_lines()[-1]

    # Decorate
    ax.set_ylabel("Empirical CDF [-]")
    ax.set_ylim((0, 1))
    ax.set_yticks([0., 0.5, 1.])
    ax.set_yticklabels(["0", chr(189)+" ", "1"])
    ylabs = ax.get_yticklabels()
    if len(ylabs)>0:
        ylabs[0].set_va("bottom")
        ylabs[2].set_va("top")

    return lines


def scattercat(ax, x, y, z, ncats=5, cuts=None, cmap="PiYG", \
                        markers=None, \
                        markersizes=None, \
                        fmt="0.2f", nval=False, eps=1e-5, \
                        alphas=1.0, *args, **kwargs):
    """ Draw a scatter plot using different colors or markersize depending
    on categories defined by z. Be careful when z has a lot of zeros,
    quantile computation may lead to non-unique category boundaries.

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
    markers : list
        Markers for each catergory
    markersizes : list
        Size of markers for each catergory
    fmnt : str
        Number format to be used in labels
    nval : bool
        Add number of points in labels
    eps : float
        Tolerance on min and max value if using cuts.
    alphas : float
        Transparency. Can be passed as vector for individual points.
    args, kwargs
        Argument sent to matplotlib.pyplot.plot command

    Returns
    -------
    plotted : dict
        Dictionary containing dictionaries for each categories index by
        category number. Each dictionary contains:
        idx :   Index of category items
        label:  Label of category
        color:  Color used for category
        line:   matplotlib.Line object
        x:      X coordinate
        y:      Y coordinate

    cats : pandas.Series
        Series containing the category number for each item
    """
    # Check inputs
    if not len(x) == len(y):
        raise ValueError("Expected x and y of same length, got "+\
                                f"len(x)={len(x)}, len(y)={len(y)}")

    if not len(x) == len(z):
        raise ValueError("Expected x and z of same length, got "+\
                                f"len(x)={len(x)}, len(z)={len(z)}")
    try:
        len(alphas)
        if not len(alphas) == len(x):
            raise ValueError("Expected x and alpha of same length, got "+\
                                f"len(x)={len(x)}, len(alphas)={len(alphas)}")
    except TypeError:
        alphas = alphas*np.ones(len(x))

    # Format categorical data
    z = pd.Series(z)

    # Check z is categorical
    # See  https://pandas.pydata.org/pandas-docs/version/0.17.0/categorical.html
    # Cell [176]
    if hasattr(z, "cat"):
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
            if cuts[0] >= z.min():
                cuts[0] = z.min()-eps
            if cuts[-1] <= z.max():
                cuts[-1] = z.max()+eps

            # Create categories
            ncats = len(cuts)-1
            cats = pd.cut(z, cuts, right=True, labels=False)
            cats[cats.isnull()] = -1
            cats = cats.astype(int)

            if len(set(cuts)) != len(cuts):
                raise ValueError(\
                        "Non-unique category boundaries :{0}".format(\
                            "/ ".join([str(u) for u in list(cuts)])))

            # Create labels
            labels = ["[{0:{fmt}}, {1:{fmt}}]".format(cuts[icat], \
                                           cuts[icat+1], fmt=fmt) \
                                               for icat in range(ncats)]
        else:
            raise ValueError("Expected ncats or cuts to be not-None")

    # Get colors for each category
    if cmap is None:
        colors = ["grey"]*ncats
    else:
        colors = cmap2colors(ncats, cmap)

    # Get size for each category
    if markers is None:
        markers = ["o"]*ncats
    else:
        errmess = f"Expected {ncats} markers, got {len(markers)}"
        assert len(markers) == ncats, errmess

    if markersizes is None:
        markersizes = [mpl.rcParams["lines.markersize"]**2]*ncats
    else:
        errmess = f"Expected {ncats} marker sizes, got {len(markersizes)}"
        assert len(markersizes) == ncats, errmess

    # Plot all categories
    plotted = {}

    for icat in range(ncats):
        # plot config
        idx = cats == icat
        if idx.sum() == 0:
            continue
        label = labels[icat]
        marker = markers[icat]
        markersize = markersizes[icat]
        col = colors[icat]

        # List of alphas
        alpha = alphas[idx]

        if np.sum(idx) > 0:
            u, v = x[idx], y[idx]
        else:
            u, v = [], []
            warnings.warn("No points falling in category "+\
                        "{0} ({1})".format(icat, label))

        # Plot category
        sc = ax.scatter(u, v, label=label, \
                        color=col, \
                        alpha=alpha, \
                        marker=marker, \
                        s=markersize, \
                        *args, **kwargs)

        # Store plotted data
        dd = {"idx": idx, \
            "label": label, \
            "color": col, \
            "scatter": sc, \
            "x": x[idx], "y": y[idx]}

        plotted[icat] = dd

    return plotted, cats


def bivarnplot(ax, xy, add_semicorr=True, namex="var 1", \
                namey="var 2", marker="o", *args, **kwargs):
    """ Bivariate normal scores Plot

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
    """
    # Select data
    idx = np.sum(np.isnan(xy), axis=1) == 0
    if np.sum(idx) < 2:
        raise ValueError("Expected at least 2 data pairs with valid"+\
                " values for both, got {}".format(np.sum(idx)))
    xy = xy[idx]

    # Compute normal standard variables and semi correlations
    unorm = np.zeros_like(xy)
    unorm[:, 0], _ = sutils.standard_normal(xy[:, 0])
    unorm[:, 1], _ = sutils.standard_normal(xy[:, 1])
    rho, eta, rho_p, rho_m = sutils.semicorr(unorm)

    # Plot
    ax.plot(unorm[:, 0], unorm[:, 1], marker, *args, **kwargs)

    line(ax, 1, 0, 0, 0, "k--", lw=0.6)
    line(ax, 0, 1, 0, 0, "k--", lw=0.6)

    # Add semi correlations
    if add_semicorr:
        text = r"$\rho$   {:5.2f}".format(rho)
        text += "\n"+r"$\eta$   {:5.2f}".format(eta)
        text += "\n"+r"$\rho^+$ {:5.2f}".format(rho_p)
        text += "\n"+r"$\rho^-$ {:5.2f}".format(rho_m)
        ax.text(0.02, 0.98, text, transform=ax.transAxes, \
                va="top", ha="left")
    # Decorate
    ax.set_xlabel("Standard normal score for {} [-]".format(namex))
    ax.set_ylabel("Standard normal score for {} [-]".format(namey))


def waterbalplot(ax, ncoeff=2.5):
    """ Background for the normalised P/PE vs Q/P adimensional
    plot. Useful to check catchment water balance.

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw the line on
    ncoeff : float
        Coefficient of the Turc-Mezentsev model.

    Returns
    -------
    tm_line : matplotlib.lines.Line2D
        Line representing the Truc-Mezentsev model.
    """
    xx = np.linspace(1e-5, 5, 1000)
    yy = np.maximum(0, 1-1./xx)
    ax.fill_between(xx, yy*0., yy, facecolor="k",
                            edgecolor="k", hatch="\\", alpha=0.2)
    ax.fill_between(xx, yy*0.+1., yy*0.+5., facecolor="k", \
                            edgecolor="k", hatch="\\", alpha=0.2)

    yy = 1-1./(1.+xx**ncoeff)**(1./ncoeff)

    lines = ax.plot(xx, yy, "k-", lw=1, \
                label=f"Turc-Mezensev (n={ncoeff:0.1f})")
    tm_line = lines[-1]
    ax.set_xlim((0, 3))
    ax.set_ylim((0, 1.1))
    ax.set_xlabel("Aridity P/PE [-]")
    ax.set_ylabel("Runoff coef Q/P [-]")

    return tm_line


def add_metadata_to_png(fimg, metadata, author=None):
    """ Add metadata to a png file

    Parameters
    ----------
    fimg : pathlib.Path or str
        Path to png file
    metadata : dict
        Dictionary {str, str} containing metadata key and value.
    add_author : bool
        Automatically add author if not provided in metadata.
    add_time : bool
        Automatically add time of creation if not provided in metadata.
    """
    # Enforce a key named "source file" to locate
    # generation script
    errmsg = "Expected metadata to have a key named 'source_file'"
    assert "source_file" in metadata, errmsg

    # Check source file exists
    source_file = metadata["source_file"]
    if isinstance(source_file, str):
        source_file = Path(source_file)
    errmsg = f"{source_file} does not exists"
    assert source_file.exists(), errmsg

    # Add metadata
    img = PngImageFile(fimg)
    fileinfo = PngInfo()
    for key, value in metadata.items():
        k = re.sub(" ", "_", str(key))
        fileinfo.add_text(k, str(value))

    # Add author and time created
    if author is None:
        author = getuser()
    fileinfo.add_text("author", author)

    fileinfo.add_text("time_created", str(datetime.now()))

    # Store in png file
    img.save(fimg, pnginfo=fileinfo)


def read_metadata_from_png(fimg):
    """ Reads metadata from a png file

    Parameters
    ----------
    fimg : pathlib.Path or str
        Path to png file

    Returns
    -------
    metadata : dict
        Dictionary {str, str} containing metadata key and value.
    """
    img = PngImageFile(fimg)
    metadata = img.text

    return metadata
