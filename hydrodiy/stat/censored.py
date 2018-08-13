import math

import numpy as np
import pandas as pd
from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal as mvt
from scipy.optimize import fmin, brent

# Small number used to detect censor data
EPS = 1e-10


def normcensloglike(y, mu, sig, censor, icens=None):
    ''' Censored log likelihood

    Parameters
    -----------
    y : numpy.ndarray
        Sample data (no nan values)
    mu : float
        Location parameter
    sig : float
        Scale parameter
    censor : float
        Censoring threshold
    icens : numpy.ndarray
        Indexes of censored data

    Returns
    -----------
    ll : float
        Censored log likelihood
    '''
    # Finds censored data
    if icens is None:
        icens = y < censor + EPS
    ncens = np.sum(icens)
    nval = len(y)

    # Censored part of the likelihood
    ll = 0.
    if ncens > 0:
        ll += ncens * norm.logcdf(censor, loc=mu, scale=sig)

    # Non-censored part of the likelihood
    if nval-ncens > 0:
        ll += np.sum(norm.logpdf(y[~icens], loc=mu, scale=sig))

    return ll


def checkdata2d(Y):
    ''' Check 2d Y array '''
    Y = np.atleast_2d(Y)
    if not Y.shape[1] == 2:
        raise ValueError('Expected Y data of shape [nx2], got {0}'.format(\
                    Y.shape))
    return Y


def censindexes(Y):
    ''' Compute censoring indexes
    Parameters
    -----------
    Y : numpy.ndarray
        2D Sample data [n x 2]

    Returns
    -----------
    icens : numpy.ndarray
        Indexes of censored data
        * 0 : no censored data
        * 1 : First variable is censored
        * 2 : Second variable is censored
        * 3 : Both variables are censored
    '''
    Y = checkdata2d(Y)

    # Individual indexes
    idx = (Y < EPS).astype(int)
    idx[:, 1] *= 2

    return np.sum(idx, axis=1)



def normcensloglike2d(Y, mu, Sig, censor, icens=None):
    ''' Bivariate normal censored log likelihood

    Parameters
    -----------
    Y : numpy.ndarray
        2D Sample data [nx2] (no nan values)
    mu : float
        Location parameter
    sig : float
        Scale parameter
    censor : float
        Censoring threshold
    icens : numpy.ndarray
        Indexes of censored data (see hydrodiy.stat.censored.censindexes)

    Returns
    -----------
    ll : float
        Censored log likelihood
    '''
    # Check data
    Y = checkdata2d(Y)
    if not mu.shape == (2, ):
        raise ValueError('Expected a mu vector of size [2], '+\
                'got {0}'.format(mu.shape))

    if not Sig.shape == (2, 2):
        raise ValueError('Expected a Sig matrix of size [2, 2], '+\
                'got {0}'.format(Sig.shape))

    # Variances and correlation
    variances = np.diag(Sig)
    corr = Sig[0, 1]/np.prod(np.sqrt(variances))

    # Finds censored data
    if icens is None:
        icens = censindexes(Y)

    # Censoring cases
    ll = 0.
    for case in range(4):
        idx = icens == case
        ncase = np.sum(idx)

        # Skip case if no data
        if ncase == 0:
            continue

        # Proceeds with log likelihood computation
        if case == 0:
            # No censoring:
            ll += mvt.logpdf(Y[idx, :], mean=mu, cov=Sig)

        elif case in [1, 2]:
            # Select which variable is censored and which one is not
            icasec = case-1
            icasenc = 2-case

            # Use conditional factoring of joint distribution
            # p(u, censor) = integ p(u, v) dv
            #              = integ p(v|u) f1(u) dv
            #              = f1(u) integ p(v|u) dv
            #              = f1(u) F2(censor|u)

            # Compute loglike for non censored variable
            # use logpdf function
            logpu = norm.logpdf(Y[idx, icasenc], \
                    loc=mu[icasenc], scale=math.sqrt(variances[icasenc]))

            # Compute loglike for censored variable
            # using conditional parameters
            mu_update = mu[icasec] \
                + Sig[0, 1]/variances[icasenc]*(Y[idx, icasenc]-mu[icasenc])
            variance_update = variances[icasec] \
                - Sig[0, 1]**2/variances[icasenc]

            # use logcdf function (careful, not logpdf here)
            logFc = norm.logcdf(censor, \
                        loc=mu_update, scale=math.sqrt(variance_update))

            ll += np.sum(logpu+logFc)

        else:
            # Compute
            # integ p(u, v) du dv
            lower = np.zeros(2)
            upper = (censor-mu)/np.sqrt(variances)
            infin = np.zeros(2)
                # <0=[-inf, +inf]
                # 0=[-inf, upper]
                # 1=[lower, +inf]
                # 2=[lower, upper]

            # Compute cdf doubled censored
            err, cdf, info = mvn.mvndst(lower, upper, infin, corr)
            ll += np.log(cdf)*ncase
            import pdb; pdb.set_trace()


    return ll


def normcensfit1d(x, censor, cst=0.375, sort=True, icens=None):
    ''' Fit a censored normal distribution using ordered stats

    Parameters
    -----------
    x : numpy.ndarray
        Sample data (no nan values)
    censor : float
        Censoring threshold
    cst : float
        Plotting position constant. Suggested values are:
        * 0.3: Value proposed by Benar and Bos-Levenbach (1953)
        * 0.375: This is Blom's value to approximate the mean of normal
                order statistics (Blom, 1958)
        * 0.3175: This is Filliben's value to approximate the mode
                of uniform order statistics (Filliben, 1975)
    sort : bool
        Perform sorting of x. Can be avoided if x is already sorted
    icens : numpy.ndarray
        Indexes of censored data

    Returns
    -----------
    mu : float
        Location parameter
    sig : float
        Scale parameter
    '''
    # Filter data
    if np.any(np.isnan(x)):
        raise ValueError('Expected no nan values in x')

    if sort:
        xs = np.sort(x)
    else:
        xs = x

    # locate censored data
    if icens is None:
        icens = xs < censor + EPS

    ncens = np.sum(icens)

    # Plotting positions
    nval = len(xs)
    ff = (np.arange(1, nval+1)-cst)/(nval+1-2*cst)

    # Linear regression against ordered values
    nocens = nval-ncens
    if nocens >= 2:
        # Standard quantiles
        qq = norm.ppf(ff[~icens])
        M = np.column_stack([np.ones(nval-ncens), qq])
        theta, _, _, _ = np.linalg.lstsq(M, xs[~icens])

    elif nocens <= 1:
        # Probability of censored data
        F0 = (nval-1.)/nval
        Q0 = norm.ppf(F0)
        # Probability of extreme value
        F1 = (nval-0.5)/nval
        Q1 = norm.ppf(F1)

        # Regression
        M = np.array([[1, Q0], [1, Q1]])
        if nocens == 1:
            theta = np.dot(np.linalg.inv(M), [censor, xs[-1]])
        else:
            theta = np.dot(np.linalg.inv(M), [censor-1., censor])

        # Case where sig < 0
        if theta[-1] < 0:
            mu = censor-1
            sig = (censor-mu)/norm.ppf(F1)
            theta = [mu, sig]

    return theta


