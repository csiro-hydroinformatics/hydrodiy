import math
import pandas as pd
import numpy as np

from itertools import product as prod

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from hydrodiy.plot import putils


def slice2d(ax, logpost, params, ip1, ip2, dval1, dval2, nval=30):
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
        zz[k, l] = logpost(pp, *logpost_args)

    # plot
    ax.contourf(xx, yy, zz, cmap='Reds')
    cs = ax.contour(xx, yy, zz, colors='grey')
    plt.clabel(cs, inline=1, fontsize=10)
    ax.plot(params[i], params[j], 'ow')
    putils.line(ax, 1, 0, 0, params[ip2], 'w--', lw=0.5)
    putils.line(ax, 0, 1, params[ip1], 0, 'w--', lw=0.5)

    ax.set_xlabel('Param {0}'.format(ip1))
    ax.set_ylabel('Param {0}'.format(ip2))


def plotchains(samples, accept):
    ''' Plot MCMC chains '''

    # Get dimensions
    nchains, nens, nparams = sample.shape

    # Initialise figure
    fig = plt.figure()
    gs = GridSpec(nparams, 2*nparams)
    axs = [[None]*(nparams+1)]*nparams

    # Data ranges
    ranges = np.zeros((nparams, 2))
    for i in range(nparams):
        smp = sample[:, :, i]
        ranges[i, 0] = smp.min()
        ranges[i, 1] = smp.max()

    # Get convergence stats
    Rc, acf = convergence(sample, nlag=1)

    # Loop through parameters
    for i in range(nparams):
        ax = plt.subplot(gs[i, :nparams])

        # Plot traces
        smp = sample[:, :, i]
        for ic in range(nchains):
            smpc = smp[ic, :]
            ar1 = acf[0, ic, i]
            ax.plot(smpc, #
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
            x = sample[:, :, j].ravel()
            y = sample[:, :, i].ravel()
            xy = np.column_stack([x, y])
            xx, yy, zz = putils.kde(xy)
            ax.contourf(xx, yy, zz, cmap='Reds')
            ax.plot(x, y, 'o', color='grey', alpha=0.3, markersize=0.5)

            # Decorate
            ax.set_xlabel('P{0}'.format(j))
            ax.set_ylabel('P{0}'.format(i))
            ax.set_xlim(ranges[j, :])
            ax.set_ylim(ranges[i, :])
            corr = np.corrcoef(xy.T)[0, 1]
            title = 'Correl P{0}/P{1}:{2:0.2f}'.format(i, j, corr)
            ax.set_title(title)
            axs[i][j] = ax

    return fig, np.array(axs)


# ----- LOG LIKELIHOOD AND POSTERIOR ----------------------
def lsnorm_loglike(params, yc, ncens, censory):

    # get parameters
    a, b, mu, sig = trans2true(params)
    LS.a = a
    LS.b = b

    # Transformed censor
    tcensory = LS.forward(censory)

    # Non censored part of the log-likelihood
    tyc = LS.forward(yc)
    jyc = LS.jacobian_det(yc)
    ll = np.sum(norm.logpdf(tyc, loc=mu, scale=sig)+np.log(jyc))

    # Censored likelihood
    if ncens>0:
        ll += norm.logcdf(tcensory, loc=mu, scale=sig)*ncens

    # Check
    if np.isnan(ll):
        isnan = np.sum(np.isnan(yc))
        raise ValueError('Log likelihood returns NaN. '+\
            'yc has {0} NaN values'.format(isnan))

    return ll


def norm_logpost(tparams, xc, ncens, censor):
    ''' Log posterior of a normal distribution with uniformative prior '''
    mu, lsig = tparams
    sig = math.exp(lsig)
    lp = np.sum(norm.logpdf(xc, loc=mu, scale=sig))

    if ncens>0:
        lp += norm.logcdf(censor, loc=mu, scale=sig)

    return lp


def lsnorm_logpost(tparams, yc, ncens, censory, sigb):
    ''' Log posterior with informative prior for the second param '''

    # Log likelihood
    ll = lsnorm_loglike(tparams, yc, ncens, censory)

    # Adds informative prior for b
    lp = ll + norm.logpdf(tparams[1], loc=0, scale=sigb)

    # Does not allow log(a) to be positive
    if tparams[0]>0:
        return -np.inf

    return lp


# ----- FITTING FUNCTIONS  ----------------------
def max_lsnorm_logpost(yc, icens, ncens, censory, sigb, \
                            *args, **kwargs):
    ''' Finds a maximum posterior for logsinh parameters '''

    nval = len(yc)+ncens
    yc = np.sort(yc)

    # Regression inputs
    u = norm.ppf(sutils.ppos(nval))
    uc = u[~icens]
    ones = np.ones(nval-ncens)

    # -- 1. Find starting point --
    # Values of a and b
    na = 30
    tavalues = np.linspace(-5, 0, na)

    nb = 30
    ff = sutils.ppos(nb)
    tbvalues = norm.ppf(ff, loc=0, scale=sigb)

    maxlogpost = -np.inf
    ini_tparams = None
    for ta, tb in prod(tavalues, tbvalues):
        # get parameters
        LS.a = math.exp(ta)
        LS.b = math.exp(tb)

        # transform data
        tyc = LS.forward(yc)

        # Set ols reg
        M = np.column_stack([ones, tyc])
        theta, _, _, _ = np.linalg.lstsq(M, uc)
        m = theta[0]
        s = theta[1]

        # Compute log post
        if s>0:
            tparams = [ta, tb, m/s, math.log(s)]
            lp = lsnorm_logpost(tparams, yc, ncens, censory, sigb)
        else:
            lp = -np.inf

        if lp>maxlogpost:
            maxlogpost = lp
            ini_tparams = tparams

    if ini_tparams is None:
        raise ValueError('Cannot identify starting point')

    # -- 2. Optimisation --
    def ofun(tparams):
        return -lsnorm_logpost(tparams, yc, ncens, censory, sigb)
    opt_tparams = fmin(ofun, ini_tparams, *args, **kwargs)

    return opt_tparams, ini_tparams


def sample4(x, censor, sigb=0.3, \
                nchains=5, nwarm=5000, nens=1000, nprint=500, nskim=10, \
                *args, **kwargs):
    ''' Infer the 4 parameters with MCMC '''

    # Rescale data
    xc, icens, ncens = cens(x, censor)
    yc, cst = rescale(xc)
    censory = censor*cst

    # -- Maximise log post with narrow informative prior for b
    #    to define proposal distribution
    start, _ = max_lsnorm_logpost(yc, icens, ncens, censory, sigb, \
                            *args, **kwargs)

    args = (yc, ncens, censory, sigb, )
    cov_jump = invhessian_approx(lsnorm_logpost, start, \
                        logpost_args=args)

    if not is_semidefinitepos(cov_jump):
        warnings.warn('Cannot use Hessian approximate. Use the diagonal')
        cov_jump = np.diag(np.diag(cov_jump))

    # --- Second sample via MCMC
    return mcmc(lsnorm_logpost, start, cov_jump, \
                        nchains, nwarm, nens, nprint, nskim, \
                        logpost_args=(yc, ncens, censory, sigb, ))



#def sample2(y, censory, \
#                nchains=5, nwarm=5000, nens=10000, \
#                *args, **kwargs):
#    ''' Infer fixed parameter transforms and sample mu and sig
#        This is what QJ calls "2 step procedure"
#    '''
#
#    # -- First, maximise log post with wide informative prior for b
#    theta = maxlogpost(y, censory, sigb=1, *args, **kwargs)
#    a, b, mu, sig = trans2true(theta)
#    LS.a = a
#    LS.b = b
#
#    # -- Second, sample mu and sig by MCMC
#
#    # .. set data
#    yc, ncens = cens(y, censory)
#    tcensory = LS.forward(censory)
#    tyc = LS.forward(yc)
#    jyc = LS.jacobian_det(yc)
#
#    # .. set log posterior probability function
#    def logpostfun(params):
#        mu = params[0]
#        sig = math.exp(params[1])
#
#        lp = np.sum(norm.logpdf(tyc, loc=mu, scale=sig))
#        lp += np.sum(np.log(jyc))
#        if ncens>0:
#            lp += ncens*norm.logcdf(tcensory, loc=mu, scale=sig)
#
#        return lp
#
#    # .. set proposal distribution ?? cov for sig
#    start = np.array([mu, math.log(sig)])
#    cov_jump = np.array([[sig, 0], [0, 1]])
#
#    # .. sample
#    sample, accept = mcmc(logpostfun, start, cov_jump, \
#                            nchains, nwarm, nens, nprint, nskim, \
#
#    aa = a * np.ones(len(sample))
#    bb = b * np.ones(len(sample))
#    sample = np.columns_stack([aa, bb, sample])
#
#    return sample, accept
#

#def ls_resample(x, censor=0., pmax=5, npriors=10000, nens=10000):
#
#    # Data scaling
#    cst = 5/x.max()
#    y = cst*x
#
#    # Sample priors
#    priors = np.random.uniform(0, pmax, size=[npriors, 4])
#
#    # first param : exp(a) prior ~ 1 except a<1 then  prior = 0
#    # - nothing to do -
#    # prior is 0 below a=1
#
#    # second param : exp(b) prior = N(0, 0.3)
#    priors[:, 1] = norm.rvs(loc=0, scale=0.3, size=npriors)
#
#    # third params : mu/sig prior ~ 1
#    # Symetric interval -5 to 5
#    priors[:, 2] = -5+priors[:, 2]*2
#
#    # fourth param : log(sig) prior ~ 1
#    # - nothing to do -
#
#    # Compute weights
#    weights = np.zeros(npriors)
#    for i in range(npriors):
#        weights[i] = loglike(priors[i, :], y, censor)
#
#    weights = weights - weights.max()
#    weights = np.exp(weights)
#    weights /= np.sum(weights)
#
#    # Resample
#    ess = 1./np.sum(weights**2)
#    kk = np.random.choice(np.arange(npriors), p=weights, size=nens)
#    sample = priors[kk, :]
#
#    return sample, ess, cst


