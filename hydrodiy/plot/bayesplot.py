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


def slice2d(ax, logpost, params, ip1, ip2, dval1, dval2, nval=30, \
    cmap='Reds', linecolor='grey'):
    ''' Plot a 2d slice of a logposterior

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
    nval : int
        Number of grid points for each parameter
    cmap : string
        Color map used to draw logposterior surface
    linecolor : string
        Color of contour lines
    '''

    # Make sure dval are positive
    dval1 = abs(dval1)
    dval2 = abs(dval2)

    # Defines grid
    x = np.linspace(params[ip1]-dval1, params[ip1]+dval1, nval)
    y = np.linspace(params[ip2]-dval2, params[ip2]+dval2, nval)
    xx, yy = np.meshgrid(x, y)

    # Compute logpost on grid
    zz = xx*np.nan
    for k, l in prod(range(nval), range(nval)):
        pp = params.copy()
        pp[ip1] = x[k]
        pp[ip2] = y[l]
        zz[k, l] = logpost(pp)

    # plot surface
    ax.contourf(xx, yy, zz, cmap=cmap)

    # plot contour lines
    original = mpl.rcParams['contour.negative_linestyle']
    mpl.rcParams['contour.negative_linestyle'] = 'solid'
    cs = ax.contour(xx, yy, zz, colors=linecolor)
    mpl.rcParams['contour.negative_linestyle'] = original
    plt.clabel(cs, inline=1, fontsize=10)

    # plot parameter point
    ax.plot(params[ip1], params[ip2], 'ow')
    putils.line(ax, 1, 0, 0, params[ip2], 'w--', lw=0.5)
    putils.line(ax, 0, 1, params[ip1], 0, 'w--', lw=0.5)

    # Decoration
    ax.set_xlabel('Param {0}'.format(ip1))
    ax.set_ylabel('Param {0}'.format(ip2))


def plotchains(fig, samples, accept):
    ''' Plot MCMC chains '''

    # Get dimensions
    nchains, nparams, nens = samples.shape

    if accept.shape != (nchains, ):
        raise ValueError('Expected dimensions of accept to '+
            'be ({0}, ), got {1}'.format(nchains, accept.shape))

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
    Rc = bayesutils.gelman(samples)
    acf = bayesutils.laggedcorr(samples, maxlag=1)

    # Loop through parameters
    for i in range(nparams):
        ax = plt.subplot(gs[i, :nparams])

        # Plot traces
        smp = samples[:, i, :]
        for ic in range(nchains):
            smpc = smp[ic, :]
            ar1 = acf[ic, i, 0]
            ax.plot(smpc, \
                lw=0.9, \
                label=r'Ch{0} $\rho_1$={1:0.2f} A={2:0.1f}%'.format(\
                    ic, ar1, accept[ic]*100))

        # Decorate
        title = 'Chains for param {0} - Rc = {1:0.5f}'.format(i, Rc[i])
        ax.set_title(title)
        ax.set_ylabel('P{0}'.format(i))
        ax.set_ylim(ranges[i, :])
        leg = ax.legend(loc=2)
        leg.get_frame().set_alpha(0.2)
        axs[i][0] = ax

        # Plot ditribution
        ax = plt.subplot(gs[i, nparams+i])
        kernel = gaussian_kde(smp.ravel())
        y = np.linspace(ranges[i, 0], ranges[i, 1], 100)
        x = kernel(y)
        ax.plot(x, y, '-')

        ax.set_ylabel('P{0}'.format(i))
        axs[i][i] = ax

        # Plot correlations
        for j in range(i+1, nparams):
            ax = plt.subplot(gs[i, nparams+j])

            # Plot kde density
            x = samples[:, j, :].ravel()
            y = samples[:, i, :].ravel()
            xy = np.column_stack([x, y])
            kd = gaussian_kde(xy.T)

            xg = np.linspace(*ranges[j, :], 30)
            yg = np.linspace(*ranges[i, :], 30)
            xx, yy = np.meshgrid(xg, yg)
            zz = kd(np.vstack([xx.ravel(), yy.ravel()]))
            zz = zz.reshape(xx.shape)
            ax.contourf(xx, yy, zz, cmap='Reds')

            # plot points
            ax.plot(x, y, 'o', color='grey', alpha=0.2, markersize=0.5)

            # Decorate
            ax.set_xlim(ranges[j, :])
            ax.set_ylim(ranges[i, :])

            corr = np.corrcoef(xy.T)[0, 1]
            title = r'$\rho$(P{0},P{1})={2:0.2f}'.format(i, j, corr)
            ax.set_title(title)
            axs[i][j] = ax

    return fig, np.array(axs)


