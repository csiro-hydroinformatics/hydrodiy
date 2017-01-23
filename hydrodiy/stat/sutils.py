import re
import math
import numpy as np
import pandas as pd

import c_hydrodiy_stat


def ppos(nval, cst=0.3):
    ''' Compute plotting position for sample of size nval

    Parameters
    -----------
    nval : int
        Sample size
    cst : float
        Plotting position constant

    Returns
    -----------
    ppos : numpy.ndarray
        Plotting postion
    '''
    return (np.arange(1, nval+1)-cst)/(nval+1-2*cst)


def acf(data, maxlag=1, biased=True, idx=None):
    ''' Sample auto-correlation function. The function computes the mean
    and standard deviation of lagged vectors independently.

    Parameters
    -----------
    data : numpy.ndarray
        Input data vector. [n] array.
    maxlag : int
        Maximum lag
    biased : bool
        Compute biased (using n as normalising constant)
        or unbiased (using n-k) estimator
    idx : numpy.ndarray
        Index to filter data points (e.g. data>value)
        By default all points are used

    Returns
    -----------
    acf_values : numpy.ndarray
        Autocorrelation function. [h] array.
    '''

    # Check inputs
    data = np.atleast_1d(data)
    nval = len(data)
    maxlag = int(maxlag)

    if not idx is None:
        idx = np.atleast_1d(idx)
        if len(idx) != nval:
            raise ValueError('Expected idx of length {0}, got {1}'.format(\
                nval, len(idx)))
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
        d1 = d1[idxk]
        d2 = d2[idxk]

        # Sample mean
        if k == 0:
            mean = np.mean(d1)

        # Covariance
        cov[k] = np.nansum((d1-mean)*(d2-mean))/(nval-k*(not biased))

    # ACF function
    acf_values = cov[1:]/cov[0]

    return acf_values


def ar1innov(alpha, innov, yini=0.):
    ''' Compute AR1 time series from innovation
    If there are nan in innov, the function produces nan, but
    the internal states a kept in memory.

    Parameters
    -----------
    alpha : float or np.ndarray
        AR1 coefficient. If a float is given, the
        value is repeated n times across the time series
    innov : numpy.ndarray
        Innovation time series. [n, p] array:
        - n is the number of time steps
        - p is the number of time series to process
    yini : float
        Initial condition

    Returns
    -----------
    outputs : numpy.ndarray
        Time series of AR1 simulations. [n, p] or [n] array
        if p=1

    Example
    -----------
    >>> nval = 100
    >>> innov1 = np.random.normal(size=nval)
    >>> data = sutils.ar1innov(0.95, innov1)
    >>> innov2 = sutils.ar1inverse(0.95, data)
    >>> np.allclose(innov1, innov2)
    True

    '''
    shape = innov.shape
    innov = np.atleast_2d(innov).astype(np.float64)

    yini = np.float64(yini)

    # Transpose 1d array
    if innov.shape[0] == 1:
        innov = innov.T

    # set the array contiguous to work with C
    if not innov.flags['C_CONTIGUOUS']:
        innov = np.ascontiguousarray(innov)

    # Set alpha
    alpha = np.atleast_1d(alpha).astype(np.float64)
    if np.prod(alpha.shape) == 1:
        alpha = np.ones(innov.shape[0]) * alpha[0]

    if alpha.shape[0] != innov.shape[0]:
        raise ValueError('Expected alpha of length {0}, got {1}'.format(\
            innov.shape[0], alpha.shape[0]))

    # initialise outputs
    outputs = np.zeros(innov.shape, np.float64)

    # Run model
    ierr = c_hydrodiy_stat.ar1innov(yini, alpha, innov, outputs)
    if ierr!=0:
        raise ValueError('ar1innov returns %d'%ierr)

    return np.reshape(outputs, shape)


def ar1inverse(alpha, inputs, yini=0):
    ''' Compute innovations from an AR1 time series
    If there are nan in inputs, the function produces nan, but
    the internal states a kept in memory.

    Parameters
    -----------
    alpha : float or np.ndarray
        AR1 coefficient. If a float is given, the
        value is repeated n times across the time series
    inputs : numpy.ndarray
        AR1 time series. [n, p] array
        - n is the number of time steps
        - p is the number of time series to process
    yini : float
        Initial condition

    Returns
    -----------
    innov : numpy.ndarray
        Time series of innovations. [n, p] array

    Example
    -----------
    >>> nval = 100
    >>> innov1 = np.random.normal(size=nval)
    >>> data = sutils.ar1innov(0.95, innov1)
    >>> innov2 = sutils.ar1inverse(0.95, data)
    >>> np.allclose(innov1, innov2)
    True

    '''
    shape = inputs.shape
    inputs = np.atleast_2d(inputs).astype(np.float64)

    # Transpose 1d array
    if inputs.shape[0] == 1:
        inputs = inputs.T

    # set the array contiguous to work with C
    if not inputs.flags['C_CONTIGUOUS']:
        inputs = np.ascontiguousarray(inputs)

    # Set alpha
    alpha = np.atleast_1d(alpha).astype(np.float64)
    if np.prod(alpha.shape) == 1:
        alpha = np.ones(inputs.shape[0]) * alpha[0]

    if alpha.shape[0] != inputs.shape[0]:
        raise ValueError('Expected alpha of length {0}, got {1}'.format(\
            inputs.shape[0], alpha.shape[0]))

    # Initialise innov
    innov = np.zeros(inputs.shape, np.float64)

    # Run model
    ierr = c_hydrodiy_stat.ar1inverse(yini, alpha, inputs, innov)
    if ierr!=0:
        raise ValueError('c_hystat.ar1inverse returns %d'%ierr)

    return np.reshape(innov, shape)


def lhs(nsamples, pmin, pmax):
    ''' Latin hypercube sampling

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

    '''
    nsamples = int(nsamples)

    # Process pmin and pmax
    pmin = np.atleast_1d(pmin).astype(np.float64)
    nparams = pmin.shape[0]

    pmax = np.atleast_1d(pmax)
    if pmax.shape[0] == 1:
        pmax = np.repeat(pmax, nparams)

    if len(pmax) != nparams:
        raise ValueError(('Expected pmax of length' + \
            ' {0}, got {1}').format(nparams, \
                len(pmax)))
    if np.any(pmax-pmin<=0):
        raise ValueError(('Expected pmax>pmin got' +\
            ' pmin={0} and pmax={1}').format(pmin, pmax))

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

