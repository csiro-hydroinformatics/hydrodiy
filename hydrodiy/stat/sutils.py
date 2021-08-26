import re
import math
import numpy as np
from scipy import linalg
import pandas as pd

from scipy.stats import norm

from hydrodiy import has_c_module

def ppos(nval, cst=0.3):
    """ Compute plotting position for sample of size nval

    Parameters
    -----------
    nval : int
        Sample size
    cst : float
        Plotting position constant in [0, 0.5]. Suggested values are:
        * 0.3: Value proposed by Benar and Bos-Levenbach (1953)
        * 0.375: This is Blom"s value to approximate the mean of normal
                order statistics (Blom, 1958)
        * 0.3175: This is Filliben"s value to approximate the mode
                of uniform order statistics (Filliben, 1975)

    Returns
    -----------
    ppos : numpy.ndarray
        Plotting postion
    """
    if cst < 0. or cst > 0.5:
        raise ValueError("Expected cst  in [0, 0.5], got {0}".format(cst))
    return (np.arange(1, nval+1)-cst)/(nval+1-2*cst)


def acf(data, maxlag=1, idx=None):
    """ Sample auto-correlation function. The function computes the mean
    and standard deviation of lagged vectors independently.

    Parameters
    -----------
    data : numpy.ndarray
        Input data vector. [n] array.
    maxlag : int
        Maximum lag
    idx : numpy.ndarray
        Index to filter data points (e.g. data>value)
        By default all points are used

    Returns
    -----------
    acf_values : numpy.ndarray
        Autocorrelation function. [h] array.
    cov : float
        Covariance
    """

    # Check inputs
    data = np.atleast_1d(data)
    nval = len(data)
    maxlag = int(maxlag)

    if not idx is None:
        idx = np.atleast_1d(idx)
        if len(idx) != nval:
            raise ValueError(f"Expected idx of length {nval}, "+\
                                f"got {len(idx)}.")
    else:
        idx = np.ones(nval).astype(bool)

    # initialise
    acf_values = np.zeros(maxlag)
    cov = np.zeros(maxlag+1)

    # loop through laghs
    for k in range(maxlag+1):
        # Lagged vectors
        d1 = data[k:]
        idx1 = idx[k:]

        if k>0:
            d2 = data[:-k]
            idx2 = idx[:-k]
        else:
            d2 = d1
            idx2 = idx1

        # apply filter
        idxk = idx1 & idx2
        nval = np.sum(idxk)
        d1 = d1[idxk]
        d2 = d2[idxk]

        # Sample mean
        if k == 0:
            mean = np.mean(d1)

        # Covariance
        cov[k] = np.nansum((d1-mean)*(d2-mean))/nval

    # ACF function
    acf_values = cov[1:]/cov[0]

    return acf_values, cov[0]


def lhs(nsamples, pmin, pmax):
    """ Latin hypercube sampling

    Parameters
    -----------
    nsamples : int
        Number of sample to draw
    pmin : list
        Lower bounds of parameters
    pmax : list
        Upper bounds of parameters

    Returns
    -----------
    samples : numpy.ndarray
        Parameter samples (nsamples x nparams)

    Example
    -----------
    >>> nparams = 5; nsamples = 3
    >>> sutils.lhs(nsamples, np.zeros(nparams), np.ones(nparams))

    """
    nsamples = int(nsamples)

    # Process pmin and pmax
    pmin = np.atleast_1d(pmin).astype(np.float64)
    nparams = pmin.shape[0]

    pmax = np.atleast_1d(pmax)
    if pmax.shape[0] == 1:
        pmax = np.repeat(pmax, nparams)

    if len(pmax) != nparams:
        raise ValueError(("Expected pmax of length" + \
            " {0}, got {1}").format(nparams, \
                len(pmax)))
    if np.any(pmax-pmin<=0):
        raise ValueError(("Expected pmax>pmin got" +\
            " pmin={0} and pmax={1}").format(pmin, pmax))

    # Initialise
    samples = np.zeros((nsamples, nparams))

    # Sample
    for i in range(nparams):
        du = float(pmax[i]-pmin[i])/nsamples
        u = np.linspace(pmin[i]+du/2, pmax[i]-du/2, nsamples)

        kk = np.random.permutation(nsamples)
        s = u[kk] + np.random.uniform(-du/2, du/2, size=nsamples)

        samples[:, i] = s

    return samples


def lhs_norm(nsamples, mean, cov):
    """ Latin hypercube sampling from a multivariate normal
    distribution.

    Parameters
    -----------
    nsamples : int
        Number of sample to draw
    mean : numpy.ndarray
        Mean vector
    cov : numpy.ndarray
        Covariance matrix

    Returns
    -----------
    samples : numpy.ndarray
        Parameter samples (nsamples x nvars)
    """
    nvars = len(mean)
    q = lhs(nsamples, [0]*nvars, [1]*nvars)
    nsmp = norm.ppf(q)
    S = linalg.cholesky(cov)
    smp = mean[:, None] + np.dot(S.T, nsmp.T)
    return smp.T


def standard_normal(x, cst=0., sorted=False, rank_method="average"):
    """ Compute normal standard variables

     Parameters
    -----------
    x : numpy.ndarray
        1D array containing data samples without nan.
    cst: float
        Constant used to compute plotting position.
        See hydrodiy.stat.sutils.ppos
    sorted : bool
        Is x data sorted or not?
    rank_method : string
        Method to compute ranks.
        See pandas.Series.rank

    Returns
    -----------
    ranks : numpy.ndarray
        Ranks within x vector
    unorm : numpy.ndarray
        Standard normal deviates
    """
    if np.any(np.isnan(x)):
        raise ValueError("Expected no nan values in x, "+\
                            "found {np.sum(np.isnan(x))))}.")
    nval = len(x)
    if sorted:
        ranks = np.arange(nval)
    else:
        ranks = pd.Series(x).rank(method=rank_method)-1

    unorm = norm.ppf((ranks+1-cst)/(nval+1-2*cst))

    return unorm, ranks


def semicorr(unorm):
    """ Compute semi correlation of standard normal transform values
    Useful to check symetry of correlations.

    Parameters
    -----------
    unorm : numpy.ndarray
        2D array containing standard normal deviates.
        See hydrodiy.stat.sutils.standard_normal

    Returns
    -----------
    rho : float
        Pearson correlation for the entire set
    eta : float
        Semi-correlation for a bi-variate normal with correlation = rho
    rho_p : float
        Pearson correlation when both urnorm are >0
    rho_m : float
        Pearson correlation when both urnorm are <0
    """
    # Check dimensions
    nval, ncols = unorm.shape
    if ncols != 2:
        raise ValueError("Expected a two columns array, "+\
                            f"got ncols={ncols}.")
    # Full correlations
    rho = np.corrcoef(unorm.T)[0, 1]

    # Theoretical semi-correlation with correlation = rho
    # See Joe (), Dependence Modelling with Copulas, Page 71, Equation 2.59
    beta0 = 0.25+math.asin(rho)/(2*math.pi)
    v10 = (1+rho)/(2*beta0*math.sqrt(2*math.pi))
    v20 = 1+rho*math.sqrt(1-rho**2)/(2*math.pi*beta0)
    v11 = (v20-1+rho**2)/rho
    eta = (v11-v10**2)/(v20-v10**2)

    # Sample semi-correlations
    idx = np.sum(unorm > 0, axis=1) == 2
    rho_p = np.corrcoef(unorm[idx].T)[0, 1]

    idx = np.sum(unorm < 0, axis=1) == 2
    rho_m = np.corrcoef(unorm[idx].T)[0, 1]

    return rho, eta, rho_p, rho_m


def pareto_front(data, orientation=1):
    """ Identify the non-dominated points in a multi-dimensional data set.

    Parameters
    -----------
    data : np.ndarray
        2D array containing the data set [nval x ncol].
    orientation : int
        Orientation of the metric:
        +1 : positively oriented (i.e. higher is better)
        -1 : negatively oriented (i.e. lower is better)

    Returns
    -----------
    is_dominated : numpy.ndarray
        Integer array indicating if the point is dominated (=1)
        or not (=0).

    Example
    -----------
    >>> nval, ncol = 100, 3
    >>> data = np.random.normal(size=(nval, 3))
    >>> sutils.pareto_front(data)
    """
    has_c_module("stat")

    orientation = np.int32(orientation)
    data = data.astype(np.float64)

    if data.ndim != 2:
        error_msg = "Expected data to be a 2 dimensional array, "+\
                        "got data.ndim={data.ndim}."
        raise ValueError(error_msg)


    # set the array contiguous to work with C
    if not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data)

    # initialise outputs
    isdominated = np.zeros(data.shape[0]).astype(np.int32)

    # Run model
    ierr = c_hydrodiy_stat.pareto_front(orientation, data, \
                                            isdominated)
    if ierr!=0:
        raise ValueError(f"c_hydrodiy_stat.pareto_front {ierr}")

    return isdominated


