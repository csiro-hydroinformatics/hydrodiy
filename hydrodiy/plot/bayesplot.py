import math
import pandas as pd
import numpy as np

from itertools import product as prod

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.stats import gaussian_kde

from hydrodiy.stat import bayesutils
from hydrodiy.plot import putils


def slice2d(ax, logpost, params, ip1, ip2, dval1, dval2, \
    scale1="abs", scale2="abs", nval=30, \
    cmap="Reds", linecolor="grey", \
    dlogpostmin=10, dlogpostmax=1e-1, \
    nlevels=5):
    """ Plot a 2d slice of a logposterior

    Parameters
    -----------
    ax : matplotlib.Axes.axes
        Axes to plot on
    logpost : function
        Log posterior. Callable function with signature logpost(params)
    params : numpy.ndarray
        Parameter values around which the slice is drawn
    ip1 : int
        First parameter index
    ip2 : int
        Second parameter index
    dval1 : float
        Amplitude of slice for the first parameter
    dval2 : float
        Amplitude of slice for the second parameter
    scale1 : str
        First parameter scale. Should be in  abs or log
    scale2 : str
        Second parameter scale. Should be in  abs or log
    nval : int
        Number of grid points for each parameter
    cmap : string
        Color map used to draw logposterior surface
    linecolor : string
        Color of contour lines
    dlogpostmin : float
        Difference between lowest contour level and max logpost value
    dlogpostmax : float
        Difference between highest contour level and max logpost value
    nlevels : int
        Number of contour lines
    """

    # Make sure parameter indexes makes sense
    ips = np.array([ip1, ip2])
    if np.any((ips<0) | (ips >= len(params))):
        raise ValueError(("Expected parameter indexes in [0, {0}[, "+\
                "got ip1={1}, ip2={2}").format(len(params), ip1, ip2))

    # Make sure dval are positive
    for idval, dval in enumerate([dval1, dval2]):
        if dval < 1e-8:
            raise ValueError(("Expected dval{0} greater than 0, "+\
                    "got {1}").format(idval+1, dval))

    # Make sure dlogpost are positive
    for ilogp, dlogpost in enumerate([dlogpostmin, dlogpostmax]):
        if dlogpost < 1e-8:
            dlogpostname = "min" if ilogp == 0 else "max"
            raise ValueError(("Expected dlogpost{0} greater than 0, "+\
                    "got {1}").format(dlogpostname, dlogpost))

    # Defines grid points
    def get_grid(pvalue, dval, scale, nval):
        if scale == "abs":
            x = pvalue + np.linspace(-dval, dval, nval)
        elif scale == "log":
            if pvalue<1e-10:
                raise ValueError("Expected parameter to be strictly "+\
                                "positive, got {0:3.3e}".format(pvalue))
            x = pvalue * np.logspace(-dval, dval, nval)
        else:
            raise ValueError("Expected scale in [abs/log], got "+scale)

        return x

    x = get_grid(params[ip1], dval1, scale1, nval)
    y = get_grid(params[ip2], dval1, scale1, nval)

    # Plotting grid
    xx, yy = np.meshgrid(x, y)

    # Compute logpost on grid
    zz = xx*np.nan
    for k, l in prod(range(nval), range(nval)):
        pp = params.copy()
        pp[ip1] = x[k]
        pp[ip2] = y[l]
        zz[k, l] = logpost(pp)

    # Compute levels
    maxzz = zz.max()
    levels = None
    if nlevels == 1:
        levels = [maxzz-dlogpostmin]
    elif nlevels>1:
        levels = list(np.linspace(maxzz-dlogpostmin, \
                        maxzz-dlogpostmax, nlevels))

    # plot surface
    ax.contourf(xx, yy, zz, cmap=cmap)

    # plot contour lines
    if nlevels > 0:
        original = mpl.rcParams["contour.negative_linestyle"]
        mpl.rcParams["contour.negative_linestyle"] = "solid"
        cs = ax.contour(xx, yy, zz, levels=levels, colors=linecolor)

        # Reformat levels
        cs.levels = ["{0:2.2e}".format(float(lev)) for lev in cs.levels]
        plt.clabel(cs, inline=1, fontsize=10)

        mpl.rcParams["contour.negative_linestyle"] = original

    # log axis
    if scale1 == "log":
        ax.set_xscale("log", nonpositive="clip")

    if scale2 == "log":
        ax.set_yscale("log", nonpositive="clip")

    # plot parameter point
    ax.plot(params[ip1], params[ip2], "ow")
    putils.line(ax, 1, 0, 0, params[ip2], "w--", lw=0.5)
    putils.line(ax, 0, 1, params[ip1], 0, "w--", lw=0.5)

    # Decoration
    ax.set_xlabel("Param {0}".format(ip1))
    ax.set_ylabel("Param {0}".format(ip2))

    return xx, yy, zz


def plotchains(fig, samples, accept, parnames=None):
    """ Plot MCMC chains """

    # Get dimensions
    if samples.ndim != 3:
        errmsg = "Expected samples to be 3D, got"+\
                    " samples.ndim={samples.ndim}."
        raise ValueError(errmsg)

    nchains, nparams, nens = samples.shape

    if accept.shape != (nchains, ):
        errmsg = f"Expected accept.shape = ({nchains},), "+\
            f" got {accept.shape}."
        raise ValueError(errmsg)

    # Parameter names
    if parnames is None:
        parnames = [f"P{i+1}" for i in range(nparams)]
    else:
        if len(parnames) != nparams:
            errmsg = f"Expected len(parnames)={nparams}, "+\
                f" got {len(parnames)}."
            raise ValueError(errmsg)

    # Initialise figure
    gs = GridSpec(nparams, 2*nparams)
    axs = [[None]*(nparams+1)]*nparams

    # Data ranges
    ranges = np.zeros((nparams, 2))
    for i in range(nparams):
        smp = samples[:, i, :]
        ranges[i, 0] = smp.min()
        ranges[i, 1] = smp.max()

    # Get convergence stats
    Rc = bayesutils.gelman_convergence(samples)
    acf = bayesutils.laggedcorr(samples, maxlag=1)

    # Loop through parameters
    lims = {}
    for i in range(nparams):
        ax = fig.add_subplot(gs[i, :nparams])

        # Plot traces
        smp = samples[:, i, :]
        for ic in range(nchains):
            smpc = smp[ic, :]
            ar1 = acf[ic, i, 0]
            A=accept[ic]*100
            ax.plot(smpc, lw=0.9, \
                label=f"Ch{ic+1} $\\rho_1$={ar1:0.2f} A={A:0.1f}%")

        # Decorate
        title = f"Chains for {parnames[i]}"
        if not np.isnan(Rc[i]):
            title += f" - Rc = {Rc[i]:0.5f}"

        ax.set_title(title)
        ax.set_ylabel(parnames[i])
        leg = ax.legend(loc=2)
        leg.get_frame().set_alpha(0.2)
        axs[i][0] = ax

        # Plot ditribution
        ax = fig.add_subplot(gs[i, nparams], sharey=axs[i][0])
        kernel = gaussian_kde(smp.ravel())
        y = np.linspace(ranges[i, 0], ranges[i, 1], 100)
        x = kernel(y)
        ax.plot(x, y, "-", label="samples pdf")

        ax.set_ylabel(parnames[i])
        ax.axis("off")
        axs[i][i+1] = ax

        # Plot correlations
        for j in range(i+1, nparams):
            ax = fig.add_subplot(gs[i, nparams+j], \
                        sharey=axs[i][0])

            # Plot kde density
            x = samples[:, j, :].ravel()
            y = samples[:, i, :].ravel()
            xy = np.column_stack([x, y])
            kd = gaussian_kde(xy.T)

            xg = np.linspace(ranges[j, 0], ranges[j, 1], 30)
            yg = np.linspace(ranges[i, 0], ranges[i, 1], 30)
            xx, yy = np.meshgrid(xg, yg)
            zz = kd(np.vstack([xx.ravel(), yy.ravel()]))
            zz = zz.reshape(xx.shape)
            ax.contourf(xx, yy, zz, cmap="Reds")

            # plot points
            ax.plot(x, y, "o", color="grey", alpha=0.2, markersize=0.5, \
                            label="samples")
            # Decorate
            ax.set_xlim(ranges[j, :])
            ax.axis("off")

            corr = np.corrcoef(xy.T)[0, 1]
            title = f"$\\rho$({parnames[i]},{parnames[j]})={corr:0.2f}"
            ax.set_title(title)
            axs[i][j+1] = ax

    return fig, np.array(axs)


