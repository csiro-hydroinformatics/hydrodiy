import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fmin, brent

# Small number used to detect censor data
EPS = 1e-10


def censloglike(y, mu, sig, censor, icens=None):
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


def censfitnorm(x, censor, cst=0.375, sort=True, icens=None):
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


